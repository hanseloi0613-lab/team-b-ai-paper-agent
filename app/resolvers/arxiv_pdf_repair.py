import json
import shutil
import time
from pathlib import Path

import httpx

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - arXiv Single PDF Repair
# ============================================================
#
# 목적:
#
# Core-100 신규 arXiv 35편 중
#
#   satellite_autonomy / 2303.08706v2
#
# 논문이 TeX parser 결과:
#
#   약 500 words
#   약 4,395 chars
#
# 로 너무 짧게 추출되어 Quality Gate를 통과하지 못했다.
#
#
# 따라서 이 한 편만:
#
# TeX
#   ↓
# PDF fallback
#   ↓
# pdf_parser.py
#   ↓
# cleaner.py
#
# 로 다시 처리한다.
#
#
# 기존 paper.tex는 삭제하지 않는다.
#
# 기존 resolution.json은 backup한 뒤
# selected_format만 PDF로 변경한다.
# ============================================================


# ============================================================
# Target
# ============================================================

TARGET_AXIS = "satellite_autonomy"
TARGET_SOURCE_ID = "2303.08706v2"


# ============================================================
# Paths
# ============================================================

PAPER_DIR = (
    PROJECT_ROOT
    / "data"
    / "tmp"
    / "arxiv_resolved"
    / TARGET_AXIS
    / TARGET_SOURCE_ID
)

RESOLUTION_FILE = (
    PAPER_DIR
    / "resolution.json"
)

RESOLUTION_BACKUP_FILE = (
    PAPER_DIR
    / "resolution_before_pdf_repair.json"
)

PDF_FILE = (
    PAPER_DIR
    / "paper.pdf"
)


# ============================================================
# arXiv URL
# ============================================================

PDF_URL = (
    f"https://arxiv.org/pdf/"
    f"{TARGET_SOURCE_ID}"
)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "TEAM-B-University-Research-Project/1.0"
    ),
    "Accept": "application/pdf,*/*",
}


# ============================================================
# Validation
# ============================================================

MIN_PDF_BYTES = 10_000


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
# PDF Validation
# ============================================================

def _validate_pdf_bytes(
    content: bytes,
) -> None:

    if not content:

        raise RuntimeError(
            "Downloaded PDF is empty."
        )

    if not content.startswith(
        b"%PDF"
    ):

        preview = (
            content[:200]
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            "Downloaded response is not a PDF.\n"
            f"Preview: {preview}"
        )

    if (
        len(content)
        < MIN_PDF_BYTES
    ):

        raise RuntimeError(
            "Downloaded PDF is suspiciously small: "
            f"{len(content)} bytes"
        )


def _existing_pdf_is_valid() -> bool:

    if not PDF_FILE.exists():

        return False

    try:

        content = (
            PDF_FILE.read_bytes()
        )

        _validate_pdf_bytes(
            content
        )

        return True

    except Exception:

        return False


# ============================================================
# Download
# ============================================================

def _download_pdf() -> dict:

    print()
    print(
        f"[DOWNLOAD] {PDF_URL}"
    )

    # --------------------------------------------------------
    # bounded retry
    #
    # arXiv에 공격적인 재시도를 하지 않는다.
    # --------------------------------------------------------

    retry_waits = [
        0,
        30,
        60,
    ]

    last_error = None

    for (
        attempt_index,
        wait_seconds,
    ) in enumerate(
        retry_waits,
        start=1,
    ):

        if wait_seconds > 0:

            print()
            print(
                f"[WAIT] "
                f"{wait_seconds} sec "
                f"before retry..."
            )

            time.sleep(
                wait_seconds
            )

        print(
            f"[HTTP] Attempt "
            f"{attempt_index}/"
            f"{len(retry_waits)}"
        )

        try:

            response = (
                httpx.get(
                    PDF_URL,
                    headers=HEADERS,
                    timeout=settings.request_timeout,
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

        # ----------------------------------------------------
        # Rate limit
        # ----------------------------------------------------

        if (
            response.status_code
            == 429
        ):

            last_error = (
                "HTTP 429 Too Many Requests"
            )

            print(
                "[HTTP] arXiv rate limit: 429"
            )

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            if (
                retry_after
                and retry_after.isdigit()
            ):

                extra_wait = int(
                    retry_after
                )

                print(
                    f"[HTTP] Retry-After: "
                    f"{extra_wait} sec"
                )

                time.sleep(
                    extra_wait
                )

            continue

        # ----------------------------------------------------
        # Other HTTP error
        # ----------------------------------------------------

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

        try:

            _validate_pdf_bytes(
                content
            )

        except Exception as exc:

            last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"[PDF] Validation failed: "
                f"{last_error}"
            )

            continue

        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        PDF_FILE.write_bytes(
            content
        )

        print(
            f"[SAVE] {PDF_FILE}"
        )

        print(
            f"[PDF] Size: "
            f"{len(content):,} bytes"
        )

        return {
            "url": (
                str(
                    response.url
                )
            ),

            "file_size": (
                len(content)
            ),

            "content_type": (
                response.headers.get(
                    "content-type"
                )
            ),
        }

    raise RuntimeError(
        "PDF download failed after "
        f"{len(retry_waits)} attempts.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# Backup Resolution
# ============================================================

def _backup_resolution() -> None:

    if not RESOLUTION_FILE.exists():

        raise FileNotFoundError(
            f"resolution.json not found: "
            f"{RESOLUTION_FILE}"
        )

    # 처음 한 번만 원본 backup
    if (
        not RESOLUTION_BACKUP_FILE.exists()
    ):

        shutil.copy2(
            RESOLUTION_FILE,
            RESOLUTION_BACKUP_FILE,
        )

        print(
            f"[BACKUP] "
            f"{RESOLUTION_BACKUP_FILE}"
        )

    else:

        print(
            "[BACKUP] Existing backup preserved."
        )


# ============================================================
# Update Resolution
# ============================================================

def _update_resolution(
    *,
    pdf_info: dict,
) -> None:

    resolution = (
        _load_json(
            RESOLUTION_FILE
        )
    )

    previous_format = (
        resolution.get(
            "selected_format"
        )
    )

    attempts = (
        resolution.get(
            "attempts",
            [],
        )
    )

    if not isinstance(
        attempts,
        list,
    ):

        attempts = []

    # --------------------------------------------------------
    # 같은 repair attempt 중복 방지
    # --------------------------------------------------------

    already_recorded = any(
        (
            isinstance(
                item,
                dict,
            )
            and item.get(
                "format"
            )
            == "pdf"
            and item.get(
                "repair"
            )
            is True
        )
        for item in attempts
    )

    if not already_recorded:

        attempts.append(
            {
                "format": (
                    "pdf"
                ),

                "url": (
                    pdf_info[
                        "url"
                    ]
                ),

                "success": (
                    True
                ),

                "repair": (
                    True
                ),

                "reason": (
                    "TeX parsed content failed "
                    "Core quality gate."
                ),
            }
        )

    # --------------------------------------------------------
    # Canonical resolution update
    # --------------------------------------------------------

    resolution[
        "success"
    ] = True

    resolution[
        "selected_format"
    ] = "pdf"

    resolution[
        "source_url"
    ] = (
        pdf_info[
            "url"
        ]
    )

    resolution[
        "local_file"
    ] = str(
        PDF_FILE
    )

    resolution[
        "file_size"
    ] = (
        pdf_info[
            "file_size"
        ]
    )

    resolution[
        "content_type"
    ] = (
        pdf_info.get(
            "content_type"
        )
    )

    resolution[
        "attempts"
    ] = attempts

    resolution[
        "repair"
    ] = {
        "reason": (
            "TeX extraction was too short "
            "for Core Quality Gate."
        ),

        "previous_selected_format": (
            previous_format
        ),

        "new_selected_format": (
            "pdf"
        ),

        "target_source_id": (
            TARGET_SOURCE_ID
        ),
    }

    _save_json(
        RESOLUTION_FILE,
        resolution,
    )

    print(
        f"[UPDATE] "
        f"{RESOLUTION_FILE}"
    )

    print(
        f"[UPDATE] selected_format: "
        f"{previous_format} -> pdf"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv Single PDF Repair"
    )

    print("=" * 70)

    print(
        f"Axis      : "
        f"{TARGET_AXIS}"
    )

    print(
        f"Source ID : "
        f"{TARGET_SOURCE_ID}"
    )

    print(
        f"Paper dir : "
        f"{PAPER_DIR}"
    )

    print(
        f"PDF URL   : "
        f"{PDF_URL}"
    )

    print("=" * 70)

    # ========================================================
    # Precheck
    # ========================================================

    if not PAPER_DIR.exists():

        raise FileNotFoundError(
            f"Paper directory not found: "
            f"{PAPER_DIR}"
        )

    if not RESOLUTION_FILE.exists():

        raise FileNotFoundError(
            f"resolution.json not found: "
            f"{RESOLUTION_FILE}"
        )

    current_resolution = (
        _load_json(
            RESOLUTION_FILE
        )
    )

    print()
    print(
        f"[CURRENT] selected_format = "
        f"{current_resolution.get('selected_format')}"
    )

    # ========================================================
    # Backup
    # ========================================================

    _backup_resolution()

    # ========================================================
    # PDF
    # ========================================================

    if _existing_pdf_is_valid():

        print()
        print(
            "[PDF] Existing valid PDF found."
        )

        content = (
            PDF_FILE.read_bytes()
        )

        pdf_info = {
            "url": (
                PDF_URL
            ),

            "file_size": (
                len(content)
            ),

            "content_type": (
                "application/pdf"
            ),
        }

    else:

        pdf_info = (
            _download_pdf()
        )

    # ========================================================
    # Update Resolution
    # ========================================================

    _update_resolution(
        pdf_info=pdf_info,
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)

    print(
        "PDF REPAIR COMPLETED"
    )

    print("=" * 70)

    print(
        f"PDF file : "
        f"{PDF_FILE}"
    )

    print(
        "Format   : pdf"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "1. Run arxiv_batch_parser"
    )

    print(
        "2. Check 35/35 parser PASS"
    )

    print(
        "3. Run cleaner"
    )

    print(
        "4. Check Quality PASS 35/35"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()