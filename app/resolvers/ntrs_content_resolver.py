import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - NASA NTRS Core-100 Content Resolver
# ============================================================
#
# 현재 상태:
#
#   Core Pilot 30          COMPLETE
#   arXiv final 50         COMPLETE
#   NTRS existing 15       COMPLETE
#   NTRS new selection 35  COMPLETE
#
#
# 이번 단계:
#
# ntrs_core100_selected.json
#             ↓
# 신규 NTRS 35편 ONLY
#             ↓
# NASA converted TXT
#             ↓ 실패
# text-native original
#             ↓ 실패
# PDF
#             ↓ 실패
# original PDF
#
#
# 출력:
#
# data/tmp/ntrs_resolved/
#   rover_autonomy/
#   onboard_ai/
#   satellite_autonomy/
#
#
# 각 논문:
#
#   metadata.json
#   resolution.json
#   paper.txt
#
# 또는:
#
#   paper.pdf
#
#
# 중요:
#
# - 기존 Pilot 15는 다시 처리하지 않는다.
# - candidate pool 전체를 처리하지 않는다.
# - selection manifest의 35편만 처리한다.
# - Resolver 35/35 PASS 전에는 Parser로 가지 않는다.
# ============================================================


RESOLVER_VERSION = "core100_v1"


# ============================================================
# Expected Selection
# ============================================================

EXPECTED_COUNTS = {
    "rover_autonomy": 12,
    "onboard_ai": 12,
    "satellite_autonomy": 11,
}

EXPECTED_TOTAL = sum(
    EXPECTED_COUNTS.values()
)


# ============================================================
# Paths
# ============================================================

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "selections"
    / "ntrs_core100_selected.json"
)


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tmp"
    / "ntrs_resolved"
)


REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)


REPORT_FILE = (
    REPORT_DIR
    / "ntrs_core100_content_resolution.json"
)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "TEAM-B-University-Research-Project/1.0"
    ),
    "Accept": "*/*",
}


# ============================================================
# Quality Thresholds
# ============================================================

MIN_TEXT_CHARS = 1500

MIN_ALPHA_RATIO = 0.35

MIN_PDF_BYTES = 10_000


# ============================================================
# Retry
# ============================================================

MAX_HTTP_ATTEMPTS = 3

RETRY_WAIT_SECONDS = [
    0,
    20,
    40,
]


# ============================================================
# Request Throttle
# ============================================================
#
# NASA에 실제 HTTP request를 보낼 때마다
# settings.request_delay만큼 간격을 보장한다.
#
# 현재 .env가 10초라면:
#
# request
#   ↓
# 최소 10초
#   ↓
# next request
#
# ============================================================

_LAST_REQUEST_AT = 0.0


# ============================================================
# JSON Helpers
# ============================================================

def _load_json(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _save_json(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================
# Selection Manifest
# ============================================================

def _load_selected_records() -> dict[
    str,
    list[dict],
]:

    payload = (
        _load_json(
            SELECTION_FILE
        )
    )

    # ========================================================
    # Version
    # ========================================================

    selection_version = (
        payload.get(
            "selection_version"
        )
    )

    if (
        selection_version
        != RESOLVER_VERSION
    ):

        raise RuntimeError(
            "Selection version mismatch.\n"
            f"Expected: {RESOLVER_VERSION}\n"
            f"Found   : {selection_version}"
        )

    # ========================================================
    # Source
    # ========================================================

    source = str(
        payload.get(
            "source",
            "",
        )
    ).lower()

    if source != "ntrs":

        raise RuntimeError(
            f"Unexpected selection source: "
            f"{source}"
        )

    # ========================================================
    # Canonical ID manifest
    # ========================================================

    selected_documents = (
        payload.get(
            "selected_documents"
        )
    )

    if not isinstance(
        selected_documents,
        dict,
    ):

        raise RuntimeError(
            "selected_documents missing "
            "from selection manifest."
        )

    # ========================================================
    # Metadata snapshots
    # ========================================================

    selected_records = (
        payload.get(
            "selected_records"
        )
    )

    if not isinstance(
        selected_records,
        dict,
    ):

        raise RuntimeError(
            "selected_records missing "
            "from selection manifest."
        )

    # ========================================================
    # Axis validation
    # ========================================================

    result = {}

    global_ids = set()

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        source_ids = (
            selected_documents.get(
                topic_axis
            )
        )

        records = (
            selected_records.get(
                topic_axis
            )
        )

        if not isinstance(
            source_ids,
            list,
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                "selected_documents is not a list."
            )

        if not isinstance(
            records,
            list,
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                "selected_records is not a list."
            )

        if (
            len(source_ids)
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count} IDs, "
                f"found {len(source_ids)}."
            )

        if (
            len(records)
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count} records, "
                f"found {len(records)}."
            )

        # ----------------------------------------------------
        # Record index
        # ----------------------------------------------------

        record_index = {}

        for record in records:

            if not isinstance(
                record,
                dict,
            ):

                raise RuntimeError(
                    f"{topic_axis}: "
                    "invalid selected record."
                )

            source_id = str(
                record.get(
                    "source_id",
                    "",
                )
            ).strip()

            if not source_id:

                raise RuntimeError(
                    f"{topic_axis}: "
                    "selected record has no source_id."
                )

            if source_id in record_index:

                raise RuntimeError(
                    f"{topic_axis}: "
                    f"duplicate record: "
                    f"{source_id}"
                )

            record_index[
                source_id
            ] = record

        # ----------------------------------------------------
        # Preserve canonical selected_documents order
        # ----------------------------------------------------

        ordered_records = []

        for raw_source_id in source_ids:

            source_id = str(
                raw_source_id
            ).strip()

            if source_id not in record_index:

                raise RuntimeError(
                    f"{topic_axis}: "
                    f"metadata snapshot missing: "
                    f"{source_id}"
                )

            if source_id in global_ids:

                raise RuntimeError(
                    "Cross-axis duplicate source_id: "
                    f"{source_id}"
                )

            global_ids.add(
                source_id
            )

            record = dict(
                record_index[
                    source_id
                ]
            )

            # -----------------------------------------------
            # Source check
            # -----------------------------------------------

            record_source = str(
                record.get(
                    "source",
                    "",
                )
            ).lower()

            if record_source != "ntrs":

                raise RuntimeError(
                    f"{source_id}: "
                    f"unexpected source "
                    f"{record_source}"
                )

            # -----------------------------------------------
            # Axis check
            # -----------------------------------------------

            record_axis = str(
                record.get(
                    "topic_axis",
                    "",
                )
            )

            if (
                record_axis
                != topic_axis
            ):

                raise RuntimeError(
                    f"{source_id}: "
                    f"axis mismatch "
                    f"{record_axis} != "
                    f"{topic_axis}"
                )

            # -----------------------------------------------
            # Full-text URL gate
            # -----------------------------------------------

            if not any(
                [
                    record.get(
                        "fulltext_url"
                    ),
                    record.get(
                        "original_url"
                    ),
                    record.get(
                        "pdf_url"
                    ),
                ]
            ):

                raise RuntimeError(
                    f"{source_id}: "
                    "no full-text candidate URL."
                )

            ordered_records.append(
                record
            )

        result[
            topic_axis
        ] = ordered_records

    # ========================================================
    # Total
    # ========================================================

    total = sum(
        len(records)
        for records
        in result.values()
    )

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} "
            f"selected NTRS records, "
            f"found {total}."
        )

    return result


# ============================================================
# Text Helpers
# ============================================================

def _normalize_text(
    text: str,
) -> str:

    text = (
        text
        .replace(
            "\x00",
            "",
        )
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    return text.strip()


def _alpha_ratio(
    text: str,
) -> float:

    if not text:

        return 0.0

    alpha_count = sum(
        1
        for char
        in text
        if char.isalpha()
    )

    return (
        alpha_count
        / len(text)
    )


def _looks_like_html(
    text: str,
) -> bool:

    head = (
        text[
            :3000
        ]
        .lower()
    )

    html_markers = (
        "<!doctype html",
        "<html",
        "<head",
        "<body",
        "<title>",
    )

    return any(
        marker in head
        for marker
        in html_markers
    )


def _looks_like_error_page(
    text: str,
) -> bool:

    head = (
        text[
            :4000
        ]
        .lower()
    )

    error_markers = (
        "404 not found",
        "403 forbidden",
        "access denied",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "request blocked",
        "too many requests",
    )

    return any(
        marker in head
        for marker
        in error_markers
    )


def _text_quality(
    text: str,
) -> tuple[
    bool,
    dict,
]:

    normalized = (
        _normalize_text(
            text
        )
    )

    char_count = len(
        normalized
    )

    word_count = len(
        normalized.split()
    )

    alpha_ratio = (
        _alpha_ratio(
            normalized
        )
    )

    html_detected = (
        _looks_like_html(
            normalized
        )
    )

    error_page = (
        _looks_like_error_page(
            normalized
        )
    )

    passed = (
        char_count
        >= MIN_TEXT_CHARS

        and alpha_ratio
        >= MIN_ALPHA_RATIO

        and not html_detected

        and not error_page
    )

    stats = {
        "char_count": (
            char_count
        ),
        "word_count": (
            word_count
        ),
        "alpha_ratio": (
            round(
                alpha_ratio,
                4,
            )
        ),
        "html_detected": (
            html_detected
        ),
        "error_page": (
            error_page
        ),
        "passed": (
            passed
        ),
    }

    return (
        passed,
        stats,
    )


# ============================================================
# URL Helpers
# ============================================================

def _url_extension(
    url: str | None,
) -> str:

    if not url:

        return ""

    parsed = (
        urlparse(
            url
        )
    )

    suffix = (
        Path(
            parsed.path
        )
        .suffix
        .lower()
    )

    return suffix


def _is_text_native_url(
    url: str | None,
) -> bool:

    if not url:

        return False

    extension = (
        _url_extension(
            url
        )
    )

    return extension in {
        ".txt",
        ".text",
        ".html",
        ".htm",
        ".xml",
        ".csv",
    }


def _is_pdf_url(
    url: str | None,
) -> bool:

    if not url:

        return False

    return (
        _url_extension(
            url
        )
        == ".pdf"
    )


# ============================================================
# Request Throttle
# ============================================================

def _wait_for_request_slot() -> None:

    global _LAST_REQUEST_AT

    now = (
        time.monotonic()
    )

    if (
        _LAST_REQUEST_AT
        <= 0
    ):

        return

    elapsed = (
        now
        - _LAST_REQUEST_AT
    )

    remaining = (
        settings.request_delay
        - elapsed
    )

    if remaining > 0:

        print(
            f"[WAIT] NASA request interval "
            f"{remaining:.1f} sec"
        )

        time.sleep(
            remaining
        )


# ============================================================
# HTTP Download
# ============================================================

def _download(
    url: str,
) -> httpx.Response:

    global _LAST_REQUEST_AT

    last_error = None

    for attempt in range(
        1,
        MAX_HTTP_ATTEMPTS + 1,
    ):

        retry_wait = (
            RETRY_WAIT_SECONDS[
                attempt - 1
            ]
        )

        if retry_wait > 0:

            print(
                f"[RETRY] Waiting "
                f"{retry_wait} sec..."
            )

            time.sleep(
                retry_wait
            )

        _wait_for_request_slot()

        print(
            f"[HTTP] Attempt "
            f"{attempt}/"
            f"{MAX_HTTP_ATTEMPTS}"
        )

        try:

            response = (
                httpx.get(
                    url,
                    headers=HEADERS,
                    timeout=(
                        settings
                        .request_timeout
                    ),
                    follow_redirects=True,
                )
            )

            _LAST_REQUEST_AT = (
                time.monotonic()
            )

        except Exception as exc:

            _LAST_REQUEST_AT = (
                time.monotonic()
            )

            last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"[HTTP] Request failed: "
                f"{last_error}"
            )

            continue

        # ====================================================
        # 429 Rate Limit
        # ====================================================

        if (
            response.status_code
            == 429
        ):

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            if (
                retry_after
                and retry_after.isdigit()
            ):

                server_wait = int(
                    retry_after
                )

                print(
                    f"[HTTP] 429 Retry-After: "
                    f"{server_wait} sec"
                )

                time.sleep(
                    server_wait
                )

            last_error = (
                "HTTP 429 Too Many Requests"
            )

            continue

        # ====================================================
        # Server Error
        # ====================================================

        if (
            500
            <= response.status_code
            <= 599
        ):

            last_error = (
                f"HTTP "
                f"{response.status_code}"
            )

            print(
                f"[HTTP] NASA server error: "
                f"{response.status_code}"
            )

            continue

        # ====================================================
        # Other HTTP
        # ====================================================

        try:

            response.raise_for_status()

        except Exception as exc:

            last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"[HTTP] Error: "
                f"{last_error}"
            )

            continue

        return response

    raise RuntimeError(
        "NASA download failed after "
        f"{MAX_HTTP_ATTEMPTS} attempts.\n"
        f"URL: {url}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# PDF Validation
# ============================================================

def _validate_pdf_bytes(
    content: bytes,
) -> tuple[
    bool,
    dict,
]:

    valid_header = (
        content.startswith(
            b"%PDF"
        )
    )

    file_size = len(
        content
    )

    passed = (
        valid_header
        and file_size
        >= MIN_PDF_BYTES
    )

    return (
        passed,
        {
            "valid_pdf_header": (
                valid_header
            ),
            "file_size": (
                file_size
            ),
            "passed": (
                passed
            ),
        },
    )


# ============================================================
# Existing Resolution Resume
# ============================================================

def _load_existing_valid_resolution(
    *,
    topic_axis: str,
    source_id: str,
    paper_dir: Path,
) -> dict | None:

    resolution_path = (
        paper_dir
        / "resolution.json"
    )

    if not resolution_path.exists():

        return None

    try:

        resolution = (
            _load_json(
                resolution_path
            )
        )

    except Exception:

        return None

    if not resolution.get(
        "success",
        False,
    ):

        return None

    if (
        str(
            resolution.get(
                "source_id",
                "",
            )
        )
        != source_id
    ):

        return None

    if (
        str(
            resolution.get(
                "topic_axis",
                "",
            )
        )
        != topic_axis
    ):

        return None

    selected_format = str(
        resolution.get(
            "selected_format",
            "",
        )
    )

    local_file_value = (
        resolution.get(
            "local_file"
        )
    )

    if not local_file_value:

        return None

    local_file = Path(
        local_file_value
    )

    if not local_file.exists():

        return None

    # ========================================================
    # Text
    # ========================================================

    if selected_format in {
        "txt",
        "original_text",
    }:

        try:

            text = (
                local_file
                .read_text(
                    encoding="utf-8",
                )
            )

        except Exception:

            return None

        passed, quality = (
            _text_quality(
                text
            )
        )

        if not passed:

            return None

        resolution[
            "quality"
        ] = quality

    # ========================================================
    # PDF
    # ========================================================

    elif selected_format == "pdf":

        try:

            content = (
                local_file
                .read_bytes()
            )

        except Exception:

            return None

        passed, quality = (
            _validate_pdf_bytes(
                content
            )
        )

        if not passed:

            return None

        resolution[
            "quality"
        ] = quality

    else:

        return None

    resolution[
        "reused_existing"
    ] = True

    return resolution


# ============================================================
# TXT Resolver
# ============================================================

def _try_fulltext_txt(
    url: str | None,
    paper_dir: Path,
) -> dict | None:

    if not url:

        return None

    print(
        f"[TRY] TXT      : "
        f"{url}"
    )

    try:

        response = (
            _download(
                url
            )
        )

    except Exception as exc:

        print(
            f"[FAIL] TXT HTTP: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    # 실제로 PDF가 돌아온 경우
    if response.content.startswith(
        b"%PDF"
    ):

        print(
            "[FAIL] TXT endpoint "
            "returned PDF bytes."
        )

        return None

    try:

        text = (
            response.text
        )

    except Exception as exc:

        print(
            f"[FAIL] TXT decode: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    text = (
        _normalize_text(
            text
        )
    )

    passed, quality = (
        _text_quality(
            text
        )
    )

    print(
        f"[TXT] "
        f"words="
        f"{quality['word_count']:,} "
        f"chars="
        f"{quality['char_count']:,} "
        f"alpha="
        f"{quality['alpha_ratio']} "
        f"pass="
        f"{quality['passed']}"
    )

    if not passed:

        print(
            "[FAIL] TXT quality "
            "check failed."
        )

        return None

    output_path = (
        paper_dir
        / "paper.txt"
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    return {
        "selected_format": (
            "txt"
        ),
        "source_url": (
            str(
                response.url
            )
        ),
        "local_file": (
            str(
                output_path
            )
        ),
        "content_type": (
            response.headers.get(
                "content-type"
            )
        ),
        "quality": (
            quality
        ),
        "success": (
            True
        ),
        "reused_existing": (
            False
        ),
    }


# ============================================================
# Text-native Original Resolver
# ============================================================

def _try_text_original(
    url: str | None,
    paper_dir: Path,
) -> dict | None:

    if not url:

        return None

    if not _is_text_native_url(
        url
    ):

        return None

    print(
        f"[TRY] ORIGINAL : "
        f"{url}"
    )

    try:

        response = (
            _download(
                url
            )
        )

    except Exception as exc:

        print(
            f"[FAIL] ORIGINAL HTTP: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    if response.content.startswith(
        b"%PDF"
    ):

        print(
            "[FAIL] Text-original URL "
            "returned PDF bytes."
        )

        return None

    try:

        text = (
            response.text
        )

    except Exception as exc:

        print(
            f"[FAIL] ORIGINAL decode: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    text = (
        _normalize_text(
            text
        )
    )

    passed, quality = (
        _text_quality(
            text
        )
    )

    print(
        f"[ORIGINAL] "
        f"words="
        f"{quality['word_count']:,} "
        f"chars="
        f"{quality['char_count']:,} "
        f"alpha="
        f"{quality['alpha_ratio']} "
        f"pass="
        f"{quality['passed']}"
    )

    if not passed:

        print(
            "[FAIL] ORIGINAL text "
            "quality check failed."
        )

        return None

    extension = (
        _url_extension(
            url
        )
    )

    if not extension:

        extension = ".txt"

    output_path = (
        paper_dir
        / (
            "paper_original"
            f"{extension}"
        )
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    return {
        "selected_format": (
            "original_text"
        ),
        "source_url": (
            str(
                response.url
            )
        ),
        "local_file": (
            str(
                output_path
            )
        ),
        "content_type": (
            response.headers.get(
                "content-type"
            )
        ),
        "quality": (
            quality
        ),
        "success": (
            True
        ),
        "reused_existing": (
            False
        ),
    }


# ============================================================
# PDF Resolver
# ============================================================

def _try_pdf(
    url: str | None,
    paper_dir: Path,
) -> dict | None:

    if not url:

        return None

    print(
        f"[TRY] PDF      : "
        f"{url}"
    )

    try:

        response = (
            _download(
                url
            )
        )

    except Exception as exc:

        print(
            f"[FAIL] PDF HTTP: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    content = (
        response.content
    )

    passed, quality = (
        _validate_pdf_bytes(
            content
        )
    )

    print(
        f"[PDF] "
        f"bytes="
        f"{quality['file_size']:,} "
        f"header="
        f"{quality['valid_pdf_header']} "
        f"pass="
        f"{quality['passed']}"
    )

    if not passed:

        print(
            "[FAIL] Downloaded content "
            "is not a usable PDF."
        )

        return None

    output_path = (
        paper_dir
        / "paper.pdf"
    )

    output_path.write_bytes(
        content
    )

    return {
        "selected_format": (
            "pdf"
        ),
        "source_url": (
            str(
                response.url
            )
        ),
        "local_file": (
            str(
                output_path
            )
        ),
        "content_type": (
            response.headers.get(
                "content-type"
            )
        ),
        "file_size": (
            len(
                content
            )
        ),
        "quality": (
            quality
        ),
        "success": (
            True
        ),
        "reused_existing": (
            False
        ),
    }


# ============================================================
# Resolution Builder
# ============================================================

def _build_success_resolution(
    *,
    topic_axis: str,
    source_id: str,
    title: str,
    result: dict,
    attempts: list[dict],
) -> dict:

    return {
        "resolver_version": (
            RESOLVER_VERSION
        ),
        "source": (
            "ntrs"
        ),
        "source_id": (
            source_id
        ),
        "topic_axis": (
            topic_axis
        ),
        "title": (
            title
        ),
        **result,
        "attempts": (
            attempts
        ),
    }


# ============================================================
# Resolve One Document
# ============================================================

def resolve_document(
    *,
    topic_axis: str,
    metadata: dict,
) -> dict:

    source_id = str(
        metadata.get(
            "source_id",
            "",
        )
    ).strip()

    title = str(
        metadata.get(
            "title",
            "",
        )
    ).strip()

    if not source_id:

        raise RuntimeError(
            f"{topic_axis}: "
            "source_id missing."
        )

    print()
    print("=" * 78)

    print(
        f"[RESOLVE] "
        f"{topic_axis} / "
        f"{source_id}"
    )

    print(
        f"[TITLE] "
        f"{title}"
    )

    print("=" * 78)

    # ========================================================
    # Paper directory
    # ========================================================

    paper_dir = (
        OUTPUT_ROOT
        / topic_axis
        / source_id
    )

    paper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Metadata snapshot
    # ========================================================

    metadata_path = (
        paper_dir
        / "metadata.json"
    )

    _save_json(
        metadata_path,
        metadata,
    )

    # ========================================================
    # Resume existing valid output
    # ========================================================

    existing_resolution = (
        _load_existing_valid_resolution(
            topic_axis=(
                topic_axis
            ),
            source_id=(
                source_id
            ),
            paper_dir=(
                paper_dir
            ),
        )
    )

    if (
        existing_resolution
        is not None
    ):

        print(
            "[REUSE] Existing valid "
            "resolved content."
        )

        print(
            f"[REUSE] Format: "
            f"{existing_resolution.get('selected_format')}"
        )

        return (
            existing_resolution
        )

    # ========================================================
    # URLs
    # ========================================================

    fulltext_url = (
        metadata.get(
            "fulltext_url"
        )
    )

    original_url = (
        metadata.get(
            "original_url"
        )
    )

    pdf_url = (
        metadata.get(
            "pdf_url"
        )
    )

    attempts = []

    # ========================================================
    # 1. NASA Converted Fulltext TXT
    # ========================================================

    if fulltext_url:

        result = (
            _try_fulltext_txt(
                fulltext_url,
                paper_dir,
            )
        )

        attempts.append(
            {
                "format": (
                    "txt"
                ),
                "url": (
                    fulltext_url
                ),
                "success": (
                    result is not None
                ),
            }
        )

        if result is not None:

            resolution = (
                _build_success_resolution(
                    topic_axis=(
                        topic_axis
                    ),
                    source_id=(
                        source_id
                    ),
                    title=(
                        title
                    ),
                    result=(
                        result
                    ),
                    attempts=(
                        attempts
                    ),
                )
            )

            _save_json(
                paper_dir
                / "resolution.json",
                resolution,
            )

            print(
                "[SUCCESS] TXT selected."
            )

            return resolution

    # ========================================================
    # 2. Text-native Original
    # ========================================================

    if (
        original_url
        and _is_text_native_url(
            original_url
        )
    ):

        result = (
            _try_text_original(
                original_url,
                paper_dir,
            )
        )

        attempts.append(
            {
                "format": (
                    "original_text"
                ),
                "url": (
                    original_url
                ),
                "success": (
                    result is not None
                ),
            }
        )

        if result is not None:

            resolution = (
                _build_success_resolution(
                    topic_axis=(
                        topic_axis
                    ),
                    source_id=(
                        source_id
                    ),
                    title=(
                        title
                    ),
                    result=(
                        result
                    ),
                    attempts=(
                        attempts
                    ),
                )
            )

            _save_json(
                paper_dir
                / "resolution.json",
                resolution,
            )

            print(
                "[SUCCESS] "
                "text-native original selected."
            )

            return resolution

    # ========================================================
    # 3. PDF candidate
    # ========================================================

    if pdf_url:

        result = (
            _try_pdf(
                pdf_url,
                paper_dir,
            )
        )

        attempts.append(
            {
                "format": (
                    "pdf"
                ),
                "url": (
                    pdf_url
                ),
                "success": (
                    result is not None
                ),
            }
        )

        if result is not None:

            resolution = (
                _build_success_resolution(
                    topic_axis=(
                        topic_axis
                    ),
                    source_id=(
                        source_id
                    ),
                    title=(
                        title
                    ),
                    result=(
                        result
                    ),
                    attempts=(
                        attempts
                    ),
                )
            )

            _save_json(
                paper_dir
                / "resolution.json",
                resolution,
            )

            print(
                "[SUCCESS] PDF selected."
            )

            return resolution

    # ========================================================
    # 4. Original itself may be PDF
    # ========================================================

    if (
        original_url
        and original_url
        != pdf_url
        and _is_pdf_url(
            original_url
        )
    ):

        result = (
            _try_pdf(
                original_url,
                paper_dir,
            )
        )

        attempts.append(
            {
                "format": (
                    "original_pdf"
                ),
                "url": (
                    original_url
                ),
                "success": (
                    result is not None
                ),
            }
        )

        if result is not None:

            result[
                "selected_format"
            ] = "pdf"

            resolution = (
                _build_success_resolution(
                    topic_axis=(
                        topic_axis
                    ),
                    source_id=(
                        source_id
                    ),
                    title=(
                        title
                    ),
                    result=(
                        result
                    ),
                    attempts=(
                        attempts
                    ),
                )
            )

            _save_json(
                paper_dir
                / "resolution.json",
                resolution,
            )

            print(
                "[SUCCESS] "
                "original PDF selected."
            )

            return resolution

    # ========================================================
    # Failure
    # ========================================================

    resolution = {
        "resolver_version": (
            RESOLVER_VERSION
        ),
        "source": (
            "ntrs"
        ),
        "source_id": (
            source_id
        ),
        "topic_axis": (
            topic_axis
        ),
        "title": (
            title
        ),
        "selected_format": (
            None
        ),
        "source_url": (
            None
        ),
        "local_file": (
            None
        ),
        "success": (
            False
        ),
        "reused_existing": (
            False
        ),
        "attempts": (
            attempts
        ),
    }

    _save_json(
        paper_dir
        / "resolution.json",
        resolution,
    )

    print(
        "[FAILED] No usable content."
    )

    return resolution


# ============================================================
# Axis Validation
# ============================================================

def _count_axis_success(
    results: list[dict],
) -> dict[
    str,
    int,
]:

    counts = {
        topic_axis: 0
        for topic_axis
        in EXPECTED_COUNTS
    }

    for result in results:

        if not result.get(
            "success",
            False,
        ):

            continue

        topic_axis = (
            result.get(
                "topic_axis"
            )
        )

        if topic_axis in counts:

            counts[
                topic_axis
            ] += 1

    return counts


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - NASA NTRS "
        "Core-100 Content Resolver"
    )

    print("=" * 78)

    print(
        f"Version        : "
        f"{RESOLVER_VERSION}"
    )

    print(
        f"Selection file : "
        f"{SELECTION_FILE}"
    )

    print(
        f"Output root    : "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Report         : "
        f"{REPORT_FILE}"
    )

    print(
        f"Request delay  : "
        f"{settings.request_delay} sec"
    )

    print()

    print(
        "Priority:"
    )

    print(
        "  1. NASA converted TXT"
    )

    print(
        "  2. text-native original"
    )

    print(
        "  3. PDF candidate"
    )

    print(
        "  4. original PDF fallback"
    )

    # ========================================================
    # Load frozen selection
    # ========================================================

    selected_records = (
        _load_selected_records()
    )

    print()
    print(
        "Frozen selection:"
    )

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        actual = len(
            selected_records[
                topic_axis
            ]
        )

        print(
            f"  "
            f"{topic_axis:22} : "
            f"{actual}"
        )

        if actual != expected_count:

            raise RuntimeError(
                f"{topic_axis}: "
                "selection count mismatch."
            )

    print(
        f"  "
        f"{'TOTAL':22} : "
        f"{sum(len(x) for x in selected_records.values())}"
    )

    # ========================================================
    # Counters
    # ========================================================

    total = 0

    success = 0

    failed = 0

    reused_count = 0

    txt_count = 0

    original_text_count = 0

    pdf_count = 0

    results = []

    # ========================================================
    # Resolve exactly selected 35
    # ========================================================

    for topic_axis in EXPECTED_COUNTS:

        documents = (
            selected_records[
                topic_axis
            ]
        )

        for metadata in documents:

            source_id = str(
                metadata.get(
                    "source_id",
                    "",
                )
            )

            total += 1

            try:

                resolution = (
                    resolve_document(
                        topic_axis=(
                            topic_axis
                        ),
                        metadata=(
                            metadata
                        ),
                    )
                )

                if resolution.get(
                    "success",
                    False,
                ):

                    success += 1

                    if resolution.get(
                        "reused_existing",
                        False,
                    ):

                        reused_count += 1

                    selected_format = (
                        resolution.get(
                            "selected_format"
                        )
                    )

                    if (
                        selected_format
                        == "txt"
                    ):

                        txt_count += 1

                    elif (
                        selected_format
                        == "original_text"
                    ):

                        original_text_count += 1

                    elif (
                        selected_format
                        == "pdf"
                    ):

                        pdf_count += 1

                else:

                    failed += 1

                results.append(
                    resolution
                )

            except Exception as exc:

                failed += 1

                print()
                print(
                    f"[ERROR] "
                    f"{topic_axis} / "
                    f"{source_id}"
                )

                print(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                results.append(
                    {
                        "resolver_version": (
                            RESOLVER_VERSION
                        ),
                        "source": (
                            "ntrs"
                        ),
                        "topic_axis": (
                            topic_axis
                        ),
                        "source_id": (
                            source_id
                        ),
                        "success": (
                            False
                        ),
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

    # ========================================================
    # Axis QA
    # ========================================================

    axis_success = (
        _count_axis_success(
            results
        )
    )

    # ========================================================
    # Report
    # ========================================================

    report = {
        "resolver_version": (
            RESOLVER_VERSION
        ),
        "selection_file": (
            str(
                SELECTION_FILE
            )
        ),
        "expected_total": (
            EXPECTED_TOTAL
        ),
        "total": (
            total
        ),
        "success": (
            success
        ),
        "failed": (
            failed
        ),
        "reused_existing": (
            reused_count
        ),
        "formats": {
            "txt": (
                txt_count
            ),
            "original_text": (
                original_text_count
            ),
            "pdf": (
                pdf_count
            ),
        },
        "axis_success": (
            axis_success
        ),
        "documents": (
            results
        ),
    }

    _save_json(
        REPORT_FILE,
        report,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 78)

    print(
        "NTRS CORE-100 CONTENT "
        "RESOLUTION COMPLETED"
    )

    print("=" * 78)

    print(
        f"Selected total : "
        f"{total}"
    )

    print(
        f"Success        : "
        f"{success}"
    )

    print(
        f"Failed         : "
        f"{failed}"
    )

    print(
        f"Reused        : "
        f"{reused_count}"
    )

    print()

    print(
        "Formats:"
    )

    print(
        f"  TXT           : "
        f"{txt_count}"
    )

    print(
        f"  Original text : "
        f"{original_text_count}"
    )

    print(
        f"  PDF           : "
        f"{pdf_count}"
    )

    print()

    print(
        "Axis QA:"
    )

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        actual_count = (
            axis_success.get(
                topic_axis,
                0,
            )
        )

        print(
            f"  "
            f"{topic_axis:22} : "
            f"{actual_count} / "
            f"{expected_count}"
        )

    print()

    print(
        f"Report:"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print("=" * 78)

    # ========================================================
    # Hard Gate
    # ========================================================

    all_axes_pass = all(
        axis_success.get(
            topic_axis,
            0,
        )
        == expected_count

        for (
            topic_axis,
            expected_count,
        )
        in EXPECTED_COUNTS.items()
    )

    if (
        total
        == EXPECTED_TOTAL
        and success
        == EXPECTED_TOTAL
        and failed
        == 0
        and all_axes_pass
    ):

        print(
            "[PASS] All 35 selected "
            "NTRS documents resolved."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Run the NTRS Core-100 "
            "parser for these 35 "
            "documents only."
        )

        print("=" * 78)

        return

    print(
        "[CHECK] Resolver Gate failed."
    )

    print()

    print(
        "Do NOT run the parser yet."
    )

    print(
        "Inspect failed NTRS IDs "
        "and repair only those documents."
    )

    print("=" * 78)

    raise RuntimeError(
        "NTRS Core-100 Resolver "
        f"Gate failed: "
        f"{success}/{EXPECTED_TOTAL} "
        "resolved."
    )


if __name__ == "__main__":
    main()