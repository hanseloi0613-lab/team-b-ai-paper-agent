import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx

from app.config import PROJECT_ROOT, settings

# ============================================================
# TEAM B - NTRS DOCX One-Paper Repair
# ============================================================
#
# Target:
#
#   onboard_ai / 20250002440
#
# Problem:
#
#   NASA converted TXT
#       -> quality gate FAIL
#
#   original
#       -> DOCX
#
#   PDF
#       -> None
#
# Therefore:
#
#   original DOCX
#       ↓
#   download
#       ↓
#   word/document.xml
#       ↓
#   extract paragraph text
#       ↓
#   paper.txt
#       ↓
#   resolution.json repair
#
#
# 중요:
#
# - 다른 34편은 건드리지 않는다.
# - threshold를 낮추지 않는다.
# - 별도 python-docx dependency를 추가하지 않는다.
# - Python standard library로 DOCX XML을 읽는다.
# ============================================================


REPAIR_VERSION = "ntrs_docx_repair_v1"

TARGET_AXIS = "onboard_ai"

TARGET_SOURCE_ID = "20250002440"


# ============================================================
# Paths
# ============================================================

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "selections"
    / "ntrs_core100_selected.json"
)


PAPER_DIR = (
    PROJECT_ROOT
    / "data"
    / "tmp"
    / "ntrs_resolved"
    / TARGET_AXIS
    / TARGET_SOURCE_ID
)


METADATA_FILE = (
    PAPER_DIR
    / "metadata.json"
)


RESOLUTION_FILE = (
    PAPER_DIR
    / "resolution.json"
)


BACKUP_RESOLUTION_FILE = (
    PAPER_DIR
    / "resolution_before_docx_repair.json"
)


DOCX_FILE = (
    PAPER_DIR
    / "paper.docx"
)


TEXT_FILE = (
    PAPER_DIR
    / "paper.txt"
)


# ============================================================
# Quality
# ============================================================

MIN_CHAR_COUNT = 5000

MIN_WORD_COUNT = 800

MIN_ALPHA_RATIO = 0.35


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "TEAM-B-University-Research-Project/1.0"
    ),
    "Accept": "*/*",
}


MAX_ATTEMPTS = 3

RETRY_DELAYS = [
    0,
    20,
    40,
]


# ============================================================
# JSON
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
# Find Target Record From Frozen Selection
# ============================================================

def _load_target_record() -> dict:

    payload = (
        _load_json(
            SELECTION_FILE
        )
    )

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

    axis_records = (
        selected_records.get(
            TARGET_AXIS
        )
    )

    if not isinstance(
        axis_records,
        list,
    ):

        raise RuntimeError(
            f"{TARGET_AXIS}: "
            "selected_records missing."
        )

    for record in axis_records:

        if not isinstance(
            record,
            dict,
        ):

            continue

        source_id = str(
            record.get(
                "source_id",
                "",
            )
        ).strip()

        if (
            source_id
            == TARGET_SOURCE_ID
        ):

            return record

    raise LookupError(
        "Target NTRS record "
        f"not found in frozen selection: "
        f"{TARGET_SOURCE_ID}"
    )


# ============================================================
# Download
# ============================================================

def _download_docx(
    url: str,
) -> tuple[
    bytes,
    str | None,
    str,
]:

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        wait_seconds = (
            RETRY_DELAYS[
                attempt - 1
            ]
        )

        if wait_seconds > 0:

            print(
                f"[RETRY] Waiting "
                f"{wait_seconds} sec..."
            )

            time.sleep(
                wait_seconds
            )

        print(
            f"[HTTP] Attempt "
            f"{attempt}/"
            f"{MAX_ATTEMPTS}"
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

        except Exception as exc:

            last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"[HTTP] Request failed: "
                f"{last_error}"
            )

            continue

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

            else:

                server_wait = max(
                    settings.request_delay,
                    30,
                )

            print(
                f"[HTTP] 429. Waiting "
                f"{server_wait} sec..."
            )

            time.sleep(
                server_wait
            )

            last_error = (
                "HTTP 429 Too Many Requests"
            )

            continue

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
                f"[HTTP] Server error: "
                f"{response.status_code}"
            )

            continue

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

        content = (
            response.content
        )

        if len(content) < 1000:

            last_error = (
                "Downloaded DOCX is "
                "suspiciously small."
            )

            print(
                f"[FAIL] {last_error}"
            )

            continue

        # DOCX는 ZIP container이다.
        if not zipfile.is_zipfile(
            io.BytesIO(
                content
            )
        ):

            last_error = (
                "Downloaded content "
                "is not a valid DOCX/ZIP."
            )

            print(
                f"[FAIL] {last_error}"
            )

            continue

        return (
            content,
            response.headers.get(
                "content-type"
            ),
            str(
                response.url
            ),
        )

    raise RuntimeError(
        "DOCX download failed after "
        f"{MAX_ATTEMPTS} attempts.\n"
        f"URL: {url}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# DOCX XML Extraction
# ============================================================

WORD_NAMESPACE = (
    "http://schemas.openxmlformats.org/"
    "wordprocessingml/2006/main"
)


def _tag(
    local_name: str,
) -> str:

    return (
        "{"
        + WORD_NAMESPACE
        + "}"
        + local_name
    )


def _extract_paragraph_text(
    paragraph: ET.Element,
) -> str:

    pieces = []

    for node in paragraph.iter():

        if node.tag == _tag(
            "t"
        ):

            if node.text:

                pieces.append(
                    node.text
                )

        elif node.tag == _tag(
            "tab"
        ):

            pieces.append(
                "\t"
            )

        elif node.tag in {
            _tag(
                "br"
            ),
            _tag(
                "cr"
            ),
        }:

            pieces.append(
                "\n"
            )

    text = "".join(
        pieces
    )

    text = text.replace(
        "\xa0",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def _extract_docx_text(
    content: bytes,
) -> str:

    with zipfile.ZipFile(
        io.BytesIO(
            content
        ),
        "r",
    ) as archive:

        names = set(
            archive.namelist()
        )

        document_xml_path = (
            "word/document.xml"
        )

        if (
            document_xml_path
            not in names
        ):

            raise RuntimeError(
                "DOCX does not contain "
                "word/document.xml."
            )

        document_xml = (
            archive.read(
                document_xml_path
            )
        )

    try:

        root = (
            ET.fromstring(
                document_xml
            )
        )

    except ET.ParseError as exc:

        raise RuntimeError(
            "Failed to parse "
            "word/document.xml."
        ) from exc

    paragraphs = []

    for paragraph in root.iter(
        _tag(
            "p"
        )
    ):

        text = (
            _extract_paragraph_text(
                paragraph
            )
        )

        if text:

            paragraphs.append(
                text
            )

    if not paragraphs:

        raise RuntimeError(
            "No text paragraphs "
            "found in DOCX."
        )

    text = "\n\n".join(
        paragraphs
    )

    text = text.replace(
        "\x00",
        "",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    return text.strip()


# ============================================================
# Quality
# ============================================================

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


def _quality(
    text: str,
) -> dict:

    char_count = len(
        text
    )

    word_count = len(
        text.split()
    )

    alpha_ratio = (
        _alpha_ratio(
            text
        )
    )

    passed = (
        char_count
        >= MIN_CHAR_COUNT

        and word_count
        >= MIN_WORD_COUNT

        and alpha_ratio
        >= MIN_ALPHA_RATIO
    )

    return {
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
            False
        ),
        "error_page": (
            False
        ),
        "passed": (
            passed
        ),
    }


# ============================================================
# Resolution Backup
# ============================================================

def _backup_resolution() -> dict:

    if not RESOLUTION_FILE.exists():

        return {}

    previous = (
        _load_json(
            RESOLUTION_FILE
        )
    )

    if not BACKUP_RESOLUTION_FILE.exists():

        shutil.copy2(
            RESOLUTION_FILE,
            BACKUP_RESOLUTION_FILE,
        )

        print(
            "[BACKUP] "
            f"{BACKUP_RESOLUTION_FILE}"
        )

    else:

        print(
            "[BACKUP] Existing backup "
            "preserved."
        )

    return previous


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - NTRS DOCX "
        "One-Paper Repair"
    )

    print("=" * 78)

    print(
        f"Target axis : "
        f"{TARGET_AXIS}"
    )

    print(
        f"Target ID   : "
        f"{TARGET_SOURCE_ID}"
    )

    print(
        f"Paper dir   : "
        f"{PAPER_DIR}"
    )

    # ========================================================
    # Frozen selection metadata
    # ========================================================

    record = (
        _load_target_record()
    )

    title = str(
        record.get(
            "title",
            "",
        )
    ).strip()

    original_url = (
        record.get(
            "original_url"
        )
    )

    fulltext_url = (
        record.get(
            "fulltext_url"
        )
    )

    pdf_url = (
        record.get(
            "pdf_url"
        )
    )

    print()
    print(
        f"[TITLE] "
        f"{title}"
    )

    print(
        f"[TXT]   "
        f"{fulltext_url}"
    )

    print(
        f"[DOCX]  "
        f"{original_url}"
    )

    print(
        f"[PDF]   "
        f"{pdf_url}"
    )

    if not original_url:

        raise RuntimeError(
            "original_url missing."
        )

    parsed_original = (
        urlparse(
            original_url
        )
    )

    if (
        Path(
            parsed_original.path
        )
        .suffix
        .lower()
        != ".docx"
    ):

        raise RuntimeError(
            "Target original file "
            "is not DOCX:\n"
            f"{original_url}"
        )

    # ========================================================
    # Directory
    # ========================================================

    PAPER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Preserve metadata snapshot
    # ========================================================

    _save_json(
        METADATA_FILE,
        record,
    )

    # ========================================================
    # Backup failed resolution
    # ========================================================

    previous_resolution = (
        _backup_resolution()
    )

    # ========================================================
    # Download DOCX
    # ========================================================

    print()
    print(
        "[DOWNLOAD] Original DOCX"
    )

    (
        docx_content,
        content_type,
        final_url,
    ) = (
        _download_docx(
            original_url
        )
    )

    print(
        f"[DOCX] Downloaded "
        f"{len(docx_content):,} bytes"
    )

    DOCX_FILE.write_bytes(
        docx_content
    )

    print(
        f"[SAVE] {DOCX_FILE}"
    )

    # ========================================================
    # Extract
    # ========================================================

    print()
    print(
        "[EXTRACT] "
        "word/document.xml"
    )

    text = (
        _extract_docx_text(
            docx_content
        )
    )

    quality = (
        _quality(
            text
        )
    )

    print(
        f"[QUALITY] "
        f"words="
        f"{quality['word_count']:,} "
        f"chars="
        f"{quality['char_count']:,} "
        f"alpha="
        f"{quality['alpha_ratio']} "
        f"pass="
        f"{quality['passed']}"
    )

    if not quality[
        "passed"
    ]:

        raise RuntimeError(
            "DOCX extracted text "
            "failed quality gate.\n"
            f"Words: "
            f"{quality['word_count']}\n"
            f"Chars: "
            f"{quality['char_count']}\n"
            f"Alpha: "
            f"{quality['alpha_ratio']}"
        )

    # ========================================================
    # Save TXT
    # ========================================================

    TEXT_FILE.write_text(
        text,
        encoding="utf-8",
    )

    print(
        f"[SAVE] {TEXT_FILE}"
    )

    # ========================================================
    # Repair resolution.json
    #
    # selected_format = txt 로 둔다.
    #
    # 이유:
    # downstream NTRS TXT parser와 그대로 호환된다.
    # 실제 provenance는 repair 필드에 DOCX라고 남긴다.
    # ========================================================

    previous_attempts = (
        previous_resolution.get(
            "attempts",
            [],
        )
    )

    if not isinstance(
        previous_attempts,
        list,
    ):

        previous_attempts = []

    attempts = [
        *previous_attempts,
        {
            "format": (
                "original_docx"
            ),
            "url": (
                original_url
            ),
            "success": (
                True
            ),
        },
    ]

    resolution = {
        "resolver_version": (
            "core100_v1"
        ),
        "source": (
            "ntrs"
        ),
        "source_id": (
            TARGET_SOURCE_ID
        ),
        "topic_axis": (
            TARGET_AXIS
        ),
        "title": (
            title
        ),
        "selected_format": (
            "txt"
        ),
        "source_url": (
            final_url
        ),
        "local_file": (
            str(
                TEXT_FILE
            )
        ),
        "content_type": (
            content_type
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
        "attempts": (
            attempts
        ),
        "repair": {
            "repair_version": (
                REPAIR_VERSION
            ),
            "reason": (
                "NASA converted TXT "
                "failed resolver quality gate "
                "and no PDF candidate existed."
            ),
            "source_format": (
                "docx"
            ),
            "original_docx_file": (
                str(
                    DOCX_FILE
                )
            ),
            "extraction_method": (
                "stdlib zipfile + "
                "word/document.xml"
            ),
        },
    }

    _save_json(
        RESOLUTION_FILE,
        resolution,
    )

    print(
        f"[SAVE] "
        f"{RESOLUTION_FILE}"
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 78)

    print(
        "NTRS DOCX REPAIR COMPLETED"
    )

    print("=" * 78)

    print(
        f"ID            : "
        f"{TARGET_SOURCE_ID}"
    )

    print(
        f"Selected fmt  : txt"
    )

    print(
        f"Source fmt    : docx"
    )

    print(
        f"Words         : "
        f"{quality['word_count']:,}"
    )

    print(
        f"Chars         : "
        f"{quality['char_count']:,}"
    )

    print(
        f"Alpha ratio   : "
        f"{quality['alpha_ratio']}"
    )

    print(
        f"Quality PASS  : "
        f"{quality['passed']}"
    )

    print()

    print(
        "[PASS] Target NTRS "
        "DOCX repaired."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Run the main "
        "ntrs_content_resolver again."
    )

    print(
        "The previous 34 documents "
        "should be reused."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()