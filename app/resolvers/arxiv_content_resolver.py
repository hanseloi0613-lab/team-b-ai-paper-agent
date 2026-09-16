import gzip
import io
import json
import re
import tarfile
import time
from pathlib import Path

import httpx

from app.config import PROJECT_ROOT, settings


# ============================================================
# Version
# ============================================================

RESOLVER_VERSION = "core100_v1"


# ============================================================
# Paths
# ============================================================
#
# Candidate metadata:
#
# data/cache/arxiv/core100_v1/
#     rover_autonomy/
#         _merged_candidates.json
#     onboard_ai/
#         _merged_candidates.json
#     satellite_autonomy/
#         _merged_candidates.json
#
# Human-QA selection:
#
# data/selections/
#     arxiv_core100_selected.json
#
# Resolved files:
#
# data/tmp/arxiv_resolved/
#     <axis>/
#         <arxiv_id>/
#             metadata.json
#             resolution.json
#             paper.html
#             or paper.tex
#             or paper.pdf
#
# 기존 Pilot 15편과 같은 root를 사용한다.
# 신규 ID 35개는 기존 15개와 중복되지 않는다.
# ============================================================

ARXIV_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "arxiv"
    / "core100_v1"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "selections"
    / "arxiv_core100_selected.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tmp"
    / "arxiv_resolved"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "arxiv_core100_content_resolution.json"
)


# ============================================================
# Expected Selection
# ============================================================
#
# 신규 arXiv 35편:
#
# rover_autonomy       12
# onboard_ai           11
# satellite_autonomy   12
#
# ============================================================

EXPECTED_COUNTS = {
    "rover_autonomy": 12,
    "onboard_ai": 11,
    "satellite_autonomy": 12,
}

EXPECTED_TOTAL = sum(
    EXPECTED_COUNTS.values()
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
# Minimum Validation
# ============================================================

MIN_HTML_BYTES = 2_000
MIN_TEX_CHARS = 1_000
MIN_PDF_BYTES = 10_000


# ============================================================
# Resume
# ============================================================
#
# True:
#
# 이미 resolution.json success가 있고
# 실제 paper.html / paper.tex / paper.pdf도 존재하면
# 다시 다운로드하지 않는다.
#
# arXiv 429 대응에도 중요.
# ============================================================

RESUME_EXISTING = True


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
# arXiv ID Helpers
# ============================================================

def _base_arxiv_id(
    source_id: str,
) -> str:
    """
    Example:

        1911.09975v1
            ↓
        1911.09975
    """

    return re.sub(
        r"v\d+$",
        "",
        source_id,
    )


# ============================================================
# Load Selection Manifest
# ============================================================

def _load_selection() -> dict[
    str,
    list[str],
]:
    """
    Human QA에서 확정한 35편만 읽는다.

    Collector의 전체 후보 pool을 resolver가
    직접 처리하지 못하게 막는 gate 역할.
    """

    payload = _load_json(
        SELECTION_FILE
    )

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
            "selection JSON does not contain "
            "'selected_documents' object."
        )

    validated = {}

    seen_base_ids = set()

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        source_ids = (
            selected_documents.get(
                topic_axis
            )
        )

        if not isinstance(
            source_ids,
            list,
        ):

            raise RuntimeError(
                f"Selection for "
                f"{topic_axis} is not a list."
            )

        cleaned_ids = []

        for source_id in source_ids:

            source_id = str(
                source_id
            ).strip()

            if not source_id:

                raise RuntimeError(
                    f"Empty source_id in "
                    f"{topic_axis}."
                )

            base_id = (
                _base_arxiv_id(
                    source_id
                )
            )

            if (
                base_id
                in seen_base_ids
            ):

                raise RuntimeError(
                    "Duplicate arXiv ID across "
                    f"selection: {source_id}"
                )

            seen_base_ids.add(
                base_id
            )

            cleaned_ids.append(
                source_id
            )

        if (
            len(cleaned_ids)
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count}, "
                f"found {len(cleaned_ids)}"
            )

        validated[
            topic_axis
        ] = cleaned_ids

    total = sum(
        len(
            ids
        )
        for ids
        in validated.values()
    )

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected total "
            f"{EXPECTED_TOTAL}, "
            f"found {total}"
        )

    declared_total = (
        payload.get(
            "selected_count"
        )
    )

    if (
        declared_total
        is not None
        and int(
            declared_total
        )
        != total
    ):

        raise RuntimeError(
            "selected_count in manifest "
            f"is {declared_total}, "
            f"but actual count is {total}."
        )

    return validated


# ============================================================
# Candidate Metadata Loading
# ============================================================

def _load_axis_candidates(
    topic_axis: str,
) -> list[dict]:

    path = (
        ARXIV_CACHE_ROOT
        / topic_axis
        / "_merged_candidates.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Core-100 arXiv candidate "
            f"file not found: {path}"
        )

    payload = _load_json(
        path
    )

    documents = (
        payload.get(
            "documents",
            [],
        )
    )

    if not isinstance(
        documents,
        list,
    ):

        raise RuntimeError(
            f"Invalid documents list: "
            f"{path}"
        )

    return documents


def _find_candidate(
    topic_axis: str,
    source_id: str,
) -> dict:

    documents = (
        _load_axis_candidates(
            topic_axis
        )
    )

    # ========================================================
    # 1. Exact version match
    # ========================================================

    for document in documents:

        candidate_id = str(
            document.get(
                "source_id",
                "",
            )
        ).strip()

        if (
            candidate_id
            == source_id
        ):

            return document

    # ========================================================
    # 2. Versionless fallback
    # ========================================================

    wanted_base = (
        _base_arxiv_id(
            source_id
        )
    )

    for document in documents:

        candidate_id = str(
            document.get(
                "source_id",
                "",
            )
        ).strip()

        if (
            _base_arxiv_id(
                candidate_id
            )
            == wanted_base
        ):

            return document

    raise LookupError(
        "Selected arXiv paper was not found "
        "in Core-100 candidate cache: "
        f"{topic_axis} / {source_id}"
    )


# ============================================================
# Candidate URLs
# ============================================================

def _build_urls(
    source_id: str,
) -> dict:

    base_id = (
        _base_arxiv_id(
            source_id
        )
    )

    return {
        "html": (
            f"https://arxiv.org/html/"
            f"{base_id}"
        ),

        "tex": (
            f"https://arxiv.org/src/"
            f"{base_id}"
        ),

        "pdf": (
            f"https://arxiv.org/pdf/"
            f"{source_id}"
        ),
    }


# ============================================================
# HTTP Request
# ============================================================

def _download(
    url: str,
) -> httpx.Response:

    response = httpx.get(
        url,
        headers=HEADERS,
        timeout=settings.request_timeout,
        follow_redirects=True,
    )

    # ========================================================
    # Rate Limit
    # ========================================================

    if response.status_code == 429:

        retry_after = (
            response.headers.get(
                "Retry-After"
            )
        )

        if (
            retry_after
            and retry_after.isdigit()
        ):

            wait_seconds = int(
                retry_after
            )

        else:

            # Retry-After가 없는 경우
            # aggressive retry는 하지 않는다.
            wait_seconds = max(
                settings.request_delay,
                15,
            )

        print()
        print(
            "[WAIT] arXiv HTTP 429"
        )

        print(
            f"[WAIT] Sleeping "
            f"{wait_seconds} sec"
        )

        time.sleep(
            wait_seconds
        )

        response = httpx.get(
            url,
            headers=HEADERS,
            timeout=settings.request_timeout,
            follow_redirects=True,
        )

        if (
            response.status_code
            == 429
        ):

            raise RuntimeError(
                "arXiv HTTP 429 is still active. "
                "Stop now and rerun later. "
                "Already completed papers will "
                "be resumed from local files."
            )

    response.raise_for_status()

    return response


# ============================================================
# HTML Validation
# ============================================================

def _valid_html(
    content: bytes,
) -> bool:

    if (
        not content
        or len(content)
        < MIN_HTML_BYTES
    ):

        return False

    try:

        head = (
            content[:5000]
            .decode(
                "utf-8",
                errors="ignore",
            )
            .lower()
        )

    except Exception:

        return False

    error_markers = (
        "404 not found",
        "page not found",
        "access denied",
        "rate exceeded",
        "too many requests",
    )

    if any(
        marker in head
        for marker
        in error_markers
    ):

        return False

    html_markers = (
        "<!doctype html",
        "<html",
        "<article",
        'class="ltx_document',
        "ltx_document",
    )

    return any(
        marker in head
        for marker
        in html_markers
    )


# ============================================================
# HTML Resolver
# ============================================================

def _try_html(
    url: str,
    paper_dir: Path,
) -> dict | None:

    print(
        f"[TRY] HTML : {url}"
    )

    try:

        response = (
            _download(
                url
            )
        )

    except Exception as exc:

        print(
            f"[FAIL] HTML HTTP: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    content = (
        response.content
    )

    if not _valid_html(
        content
    ):

        print(
            "[FAIL] HTML validation failed."
        )

        return None

    output_path = (
        paper_dir
        / "paper.html"
    )

    output_path.write_bytes(
        content
    )

    return {
        "selected_format": "html",

        "source_url": (
            url
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
    }


# ============================================================
# TeX Helpers
# ============================================================

def _decode_tex_bytes(
    content: bytes,
) -> str | None:

    encodings = (
        "utf-8",
        "utf-8-sig",
        "latin-1",
    )

    for encoding in encodings:

        try:

            text = (
                content.decode(
                    encoding
                )
            )

        except UnicodeDecodeError:

            continue

        if (
            "\\documentclass"
            in text
            or "\\begin{document}"
            in text
        ):

            return text

    return None


def _choose_main_tex(
    members: list[
        tuple[
            str,
            bytes,
        ]
    ],
) -> tuple[
    str,
    str,
] | None:

    candidates = []

    for (
        name,
        content,
    ) in members:

        if not (
            name.lower()
            .endswith(
                ".tex"
            )
        ):

            continue

        text = (
            _decode_tex_bytes(
                content
            )
        )

        if not text:

            continue

        basename = (
            Path(
                name
            )
            .name
            .lower()
        )

        score = len(
            text
        )

        if basename in {
            "main.tex",
            "paper.tex",
            "article.tex",
            "manuscript.tex",
            "ms.tex",
        }:

            score += (
                1_000_000
            )

        if (
            "\\documentclass"
            in text
        ):

            score += (
                500_000
            )

        if (
            "\\begin{document}"
            in text
        ):

            score += (
                250_000
            )

        candidates.append(
            (
                score,
                name,
                text,
            )
        )

    if not candidates:

        return None

    candidates.sort(
        reverse=True,
        key=lambda item: item[0],
    )

    (
        _,
        name,
        text,
    ) = candidates[0]

    return (
        name,
        text,
    )


def _extract_tex_from_source(
    content: bytes,
) -> tuple[
    str,
    str,
] | None:

    # ========================================================
    # 1. tar / tar.gz
    # ========================================================

    try:

        buffer = io.BytesIO(
            content
        )

        with tarfile.open(
            fileobj=buffer,
            mode="r:*",
        ) as archive:

            members = []

            for member in (
                archive.getmembers()
            ):

                if not (
                    member.isfile()
                ):

                    continue

                if not (
                    member.name
                    .lower()
                    .endswith(
                        ".tex"
                    )
                ):

                    continue

                extracted = (
                    archive.extractfile(
                        member
                    )
                )

                if (
                    extracted
                    is None
                ):

                    continue

                members.append(
                    (
                        member.name,
                        extracted.read(),
                    )
                )

            chosen = (
                _choose_main_tex(
                    members
                )
            )

            if (
                chosen
                is not None
            ):

                return chosen

    except (
        tarfile.TarError,
        EOFError,
    ):

        pass

    # ========================================================
    # 2. gzip single TeX
    # ========================================================

    try:

        decompressed = (
            gzip.decompress(
                content
            )
        )

        text = (
            _decode_tex_bytes(
                decompressed
            )
        )

        if text:

            return (
                "paper.tex",
                text,
            )

    except (
        OSError,
        EOFError,
    ):

        pass

    # ========================================================
    # 3. plain TeX
    # ========================================================

    text = (
        _decode_tex_bytes(
            content
        )
    )

    if text:

        return (
            "paper.tex",
            text,
        )

    return None


# ============================================================
# TeX Resolver
# ============================================================

def _try_tex(
    url: str,
    paper_dir: Path,
) -> dict | None:

    print(
        f"[TRY] TeX  : {url}"
    )

    try:

        response = (
            _download(
                url
            )
        )

    except Exception as exc:

        print(
            f"[FAIL] TeX HTTP: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    content = (
        response.content
    )

    if not content:

        print(
            "[FAIL] Empty TeX source."
        )

        return None

    extracted = (
        _extract_tex_from_source(
            content
        )
    )

    if (
        extracted
        is None
    ):

        print(
            "[FAIL] No usable "
            "TeX source found."
        )

        return None

    (
        original_name,
        text,
    ) = extracted

    if (
        len(
            text
        )
        < MIN_TEX_CHARS
    ):

        print(
            "[FAIL] TeX source "
            "too short."
        )

        return None

    # Raw source package
    source_path = (
        paper_dir
        / "source_package.bin"
    )

    source_path.write_bytes(
        content
    )

    # Parser-facing canonical file
    tex_path = (
        paper_dir
        / "paper.tex"
    )

    tex_path.write_text(
        text,
        encoding="utf-8",
    )

    return {
        "selected_format": "tex",

        "source_url": (
            url
        ),

        "local_file": (
            str(
                tex_path
            )
        ),

        "source_package": (
            str(
                source_path
            )
        ),

        "source_tex_name": (
            original_name
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
    }


# ============================================================
# PDF Resolver
# ============================================================

def _try_pdf(
    url: str,
    paper_dir: Path,
) -> dict | None:

    print(
        f"[TRY] PDF  : {url}"
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

    if not content.startswith(
        b"%PDF"
    ):

        print(
            "[FAIL] Response is "
            "not a PDF."
        )

        return None

    if (
        len(content)
        < MIN_PDF_BYTES
    ):

        print(
            "[FAIL] PDF is "
            "suspiciously small."
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
        "selected_format": "pdf",

        "source_url": (
            url
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
    }


# ============================================================
# Existing Resolution / Resume
# ============================================================

def _resume_existing(
    paper_dir: Path,
    source_id: str,
) -> dict | None:

    if not RESUME_EXISTING:

        return None

    resolution_path = (
        paper_dir
        / "resolution.json"
    )

    if not (
        resolution_path.exists()
    ):

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

    existing_id = str(
        resolution.get(
            "source_id",
            "",
        )
    ).strip()

    if (
        _base_arxiv_id(
            existing_id
        )
        !=
        _base_arxiv_id(
            source_id
        )
    ):

        return None

    selected_format = (
        resolution.get(
            "selected_format"
        )
    )

    expected_file = None

    if (
        selected_format
        == "html"
    ):

        expected_file = (
            paper_dir
            / "paper.html"
        )

    elif (
        selected_format
        == "tex"
    ):

        expected_file = (
            paper_dir
            / "paper.tex"
        )

    elif (
        selected_format
        == "pdf"
    ):

        expected_file = (
            paper_dir
            / "paper.pdf"
        )

    if (
        expected_file
        is None
        or not expected_file.exists()
    ):

        return None

    print(
        f"[RESUME] Existing "
        f"{selected_format.upper()} "
        f"result reused."
    )

    resolution[
        "_resumed"
    ] = True

    return resolution


# ============================================================
# Save Resolution
# ============================================================

def _save_resolution(
    *,
    paper_dir: Path,
    topic_axis: str,
    source_id: str,
    title: str,
    result: dict,
    attempts: list[dict],
) -> dict:

    resolution = {
        "resolver_version": (
            RESOLVER_VERSION
        ),

        "selection_file": (
            str(
                SELECTION_FILE
            )
        ),

        "source": "arxiv",

        "source_id": (
            source_id
        ),

        "topic_axis": (
            topic_axis
        ),

        "title": (
            title
        ),

        "success": True,

        **result,

        "attempts": (
            attempts
        ),
    }

    _save_json(
        paper_dir
        / "resolution.json",
        resolution,
    )

    return resolution


# ============================================================
# Resolve One Paper
# ============================================================

def resolve_document(
    topic_axis: str,
    source_id: str,
) -> dict:

    print()
    print("=" * 70)

    print(
        f"[RESOLVE] "
        f"{topic_axis} / "
        f"{source_id}"
    )

    print("=" * 70)

    metadata = (
        _find_candidate(
            topic_axis,
            source_id,
        )
    )

    candidate_source_id = str(
        metadata.get(
            "source_id",
            source_id,
        )
    ).strip()

    title = str(
        metadata.get(
            "title",
            "",
        )
    ).strip()

    print(
        f"[TITLE] {title}"
    )

    paper_dir = (
        OUTPUT_ROOT
        / topic_axis
        / candidate_source_id
    )

    paper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Resume
    # ========================================================

    existing = (
        _resume_existing(
            paper_dir,
            candidate_source_id,
        )
    )

    if (
        existing
        is not None
    ):

        return existing

    # ========================================================
    # Metadata snapshot
    # ========================================================

    _save_json(
        paper_dir
        / "metadata.json",
        metadata,
    )

    urls = (
        _build_urls(
            candidate_source_id
        )
    )

    attempts = []

    # ========================================================
    # 1. HTML
    # ========================================================

    result = (
        _try_html(
            urls[
                "html"
            ],
            paper_dir,
        )
    )

    attempts.append(
        {
            "format": "html",

            "url": (
                urls[
                    "html"
                ]
            ),

            "success": (
                result
                is not None
            ),
        }
    )

    if (
        result
        is not None
    ):

        resolution = (
            _save_resolution(
                paper_dir=paper_dir,
                topic_axis=topic_axis,
                source_id=candidate_source_id,
                title=title,
                result=result,
                attempts=attempts,
            )
        )

        print(
            "[SUCCESS] HTML selected."
        )

        return resolution

    time.sleep(
        settings.request_delay
    )

    # ========================================================
    # 2. TeX / LaTeX
    # ========================================================

    result = (
        _try_tex(
            urls[
                "tex"
            ],
            paper_dir,
        )
    )

    attempts.append(
        {
            "format": "tex",

            "url": (
                urls[
                    "tex"
                ]
            ),

            "success": (
                result
                is not None
            ),
        }
    )

    if (
        result
        is not None
    ):

        resolution = (
            _save_resolution(
                paper_dir=paper_dir,
                topic_axis=topic_axis,
                source_id=candidate_source_id,
                title=title,
                result=result,
                attempts=attempts,
            )
        )

        print(
            "[SUCCESS] TeX selected."
        )

        return resolution

    time.sleep(
        settings.request_delay
    )

    # ========================================================
    # 3. PDF Fallback
    # ========================================================

    result = (
        _try_pdf(
            urls[
                "pdf"
            ],
            paper_dir,
        )
    )

    attempts.append(
        {
            "format": "pdf",

            "url": (
                urls[
                    "pdf"
                ]
            ),

            "success": (
                result
                is not None
            ),
        }
    )

    if (
        result
        is not None
    ):

        resolution = (
            _save_resolution(
                paper_dir=paper_dir,
                topic_axis=topic_axis,
                source_id=candidate_source_id,
                title=title,
                result=result,
                attempts=attempts,
            )
        )

        print(
            "[SUCCESS] PDF selected."
        )

        return resolution

    # ========================================================
    # Failure
    # ========================================================

    resolution = {
        "resolver_version": (
            RESOLVER_VERSION
        ),

        "source": "arxiv",

        "source_id": (
            candidate_source_id
        ),

        "topic_axis": (
            topic_axis
        ),

        "title": (
            title
        ),

        "success": False,

        "selected_format": None,

        "source_url": None,

        "local_file": None,

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
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv Core-100 "
        "Content Resolver"
    )

    print("=" * 70)

    print(
        f"Version        : "
        f"{RESOLVER_VERSION}"
    )

    print(
        f"Selection file : "
        f"{SELECTION_FILE}"
    )

    print(
        f"Candidate root : "
        f"{ARXIV_CACHE_ROOT}"
    )

    print(
        f"Output root    : "
        f"{OUTPUT_ROOT}"
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
        "1. HTML"
    )

    print(
        "2. TeX / LaTeX source"
    )

    print(
        "3. PDF fallback"
    )

    # ========================================================
    # Selection
    # ========================================================

    try:

        selected_documents = (
            _load_selection()
        )

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            "SELECTION VALIDATION FAILED"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return

    print()
    print(
        "Selected documents:"
    )

    for (
        topic_axis,
        source_ids,
    ) in selected_documents.items():

        print(
            f"  {topic_axis:22} : "
            f"{len(source_ids)}"
        )

    print(
        f"  {'TOTAL':22} : "
        f"{sum(len(v) for v in selected_documents.values())}"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    success_count = 0
    failed_count = 0

    html_count = 0
    tex_count = 0
    pdf_count = 0
    resumed_count = 0

    results = []

    # ========================================================
    # Resolve Human-QA Selected 35
    # ========================================================

    for (
        topic_axis,
        source_ids,
    ) in selected_documents.items():

        for source_id in source_ids:

            total += 1

            try:

                resolution = (
                    resolve_document(
                        topic_axis,
                        source_id,
                    )
                )

                selected_format = (
                    resolution.get(
                        "selected_format"
                    )
                )

                if (
                    resolution.get(
                        "_resumed"
                    )
                ):

                    resumed_count += 1

                if selected_format:

                    success_count += 1

                    if (
                        selected_format
                        == "html"
                    ):

                        html_count += 1

                    elif (
                        selected_format
                        == "tex"
                    ):

                        tex_count += 1

                    elif (
                        selected_format
                        == "pdf"
                    ):

                        pdf_count += 1

                else:

                    failed_count += 1

                # report 전용 field 제거
                report_item = dict(
                    resolution
                )

                report_item.pop(
                    "_resumed",
                    None,
                )

                results.append(
                    report_item
                )

            except Exception as exc:

                failed_count += 1

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
                        "source": (
                            "arxiv"
                        ),

                        "topic_axis": (
                            topic_axis
                        ),

                        "source_id": (
                            source_id
                        ),

                        "success": False,

                        "selected_format": (
                            None
                        ),

                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

            # 성공/실패 논문 사이에서도
            # arXiv 서버에 공격적으로 요청하지 않는다.
            time.sleep(
                settings.request_delay
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

        "total": (
            total
        ),

        "success": (
            success_count
        ),

        "failed": (
            failed_count
        ),

        "resumed": (
            resumed_count
        ),

        "formats": {
            "html": (
                html_count
            ),

            "tex": (
                tex_count
            ),

            "pdf": (
                pdf_count
            ),
        },

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
    print("=" * 70)

    print(
        "ARXIV CORE-100 CONTENT "
        "RESOLUTION COMPLETED"
    )

    print("=" * 70)

    print(
        f"Total   : "
        f"{total}"
    )

    print(
        f"Success : "
        f"{success_count}"
    )

    print(
        f"Failed  : "
        f"{failed_count}"
    )

    print(
        f"Resumed : "
        f"{resumed_count}"
    )

    print(
        f"HTML    : "
        f"{html_count}"
    )

    print(
        f"TeX     : "
        f"{tex_count}"
    )

    print(
        f"PDF     : "
        f"{pdf_count}"
    )

    print(
        f"Report  : "
        f"{REPORT_FILE}"
    )

    print("=" * 70)

    if (
        success_count
        == EXPECTED_TOTAL
        and failed_count
        == 0
    ):

        print(
            "[PASS] All 35 selected "
            "arXiv papers resolved."
        )

    else:

        print(
            "[CHECK] Some selected papers "
            "still need resolution."
        )

        print(
            "Rerun later. Successful "
            "documents will be resumed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()