import html
import io
import json
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from app.config import PROJECT_ROOT, settings


# ============================================================
# Paths
# ============================================================

NTRS_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "ntrs"
    / "v2"
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
    / "ntrs_content_resolution.json"
)


# ============================================================
# Selected Pilot Documents
# ============================================================

SELECTED_DOCUMENTS = {

    "rover_autonomy": [
        "19950017272",
        "19950017269",
        "19900016232",
        "20190001760",
        "19910011338",
    ],

    "onboard_ai": [
        "20230014588",
        "20240011788",
        "20240011707",
        "20240012527",
        "20190002469",
    ],

    "satellite_autonomy": [
        "20210020739",
        "20190004938",
        "20230004265",
        "20160011976",
        "20050157886",
    ],
}


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
MIN_DOCX_BYTES = 5_000


# ============================================================
# JSON Helpers
# ============================================================

def _load_json(
    path: Path,
) -> dict:

    return json.loads(
        path.read_text(
            encoding="utf-8"
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
# Basic Text Helpers
# ============================================================

def _normalize_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = text.replace(
        "\ufeff",
        "",
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

    # non-breaking space
    text = text.replace(
        "\xa0",
        " ",
    )

    # 줄 내부의 과도한 공백만 정리
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # 빈 줄 과다 반복 정리
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
        for char in text
        if char.isalpha()
    )

    return (
        alpha_count
        / len(text)
    )


# ============================================================
# HTML / Error Detection
# ============================================================

def _looks_like_html(
    text: str,
) -> bool:

    if not text:
        return False

    head = (
        text[:4000]
        .lower()
    )

    html_markers = (
        "<!doctype html",
        "<html",
        "<head",
        "<body",
        "<title",
        "<div",
        "<p>",
        "<p ",
    )

    return any(
        marker in head
        for marker in html_markers
    )


def _looks_like_error_page(
    text: str,
) -> bool:

    if not text:
        return False

    head = (
        text[:5000]
        .lower()
    )

    error_markers = (
        "404 not found",
        "403 forbidden",
        "access denied",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "request blocked",
        "page not found",
        "not authorized",
    )

    return any(
        marker in head
        for marker in error_markers
    )


# ============================================================
# HTML/XML-ish Text Cleanup
# ============================================================

def _strip_markup(
    text: str,
) -> str:
    """
    NASA *.txt 변환 파일 안에 HTML/XML markup이
    남아 있는 경우 markup만 제거해서 본문을 살린다.

    중요한 점:
    HTML 흔적이 있다는 이유만으로 논문 TXT를
    바로 폐기하지 않는다.
    """

    if not text:
        return ""

    cleaned = text

    # script/style 제거
    cleaned = re.sub(
        r"(?is)<script\b.*?</script>",
        " ",
        cleaned,
    )

    cleaned = re.sub(
        r"(?is)<style\b.*?</style>",
        " ",
        cleaned,
    )

    # block 태그는 줄바꿈으로
    cleaned = re.sub(
        r"(?i)</?(?:p|div|section|article|"
        r"h[1-6]|li|tr|table|br|hr)\b[^>]*>",
        "\n",
        cleaned,
    )

    # 나머지 markup 제거
    cleaned = re.sub(
        r"(?s)<[^>]+>",
        " ",
        cleaned,
    )

    # HTML entity 복원
    cleaned = html.unescape(
        cleaned
    )

    return _normalize_text(
        cleaned
    )


# ============================================================
# Quality Evaluation
# ============================================================

def _evaluate_text(
    text: str,
) -> tuple[
    bool,
    str,
    dict,
]:
    """
    반환:
        passed
        prepared_text
        quality stats

    HTML/XML 흔적이 있어도 논문 텍스트 자체가
    충분하면 markup을 제거하고 다시 평가한다.
    """

    raw_normalized = (
        _normalize_text(
            text
        )
    )

    raw_html_detected = (
        _looks_like_html(
            raw_normalized
        )
    )

    raw_error_page = (
        _looks_like_error_page(
            raw_normalized
        )
    )

    transformation = "none"

    prepared_text = (
        raw_normalized
    )

    # --------------------------------------------------------
    # HTML/XML 흔적이 있을 경우
    # markup 제거 후 실제 텍스트를 다시 평가
    # --------------------------------------------------------

    if (
        raw_html_detected
        and not raw_error_page
    ):

        prepared_text = (
            _strip_markup(
                raw_normalized
            )
        )

        transformation = (
            "markup_stripped"
        )

    char_count = len(
        prepared_text
    )

    alpha_ratio = (
        _alpha_ratio(
            prepared_text
        )
    )

    final_error_page = (
        _looks_like_error_page(
            prepared_text
        )
    )

    reasons = []

    if (
        char_count
        < MIN_TEXT_CHARS
    ):

        reasons.append(
            "too_short"
        )

    if (
        alpha_ratio
        < MIN_ALPHA_RATIO
    ):

        reasons.append(
            "low_alpha_ratio"
        )

    if raw_error_page:

        reasons.append(
            "raw_error_page"
        )

    if final_error_page:

        reasons.append(
            "final_error_page"
        )

    passed = (
        len(reasons)
        == 0
    )

    stats = {
        "raw_char_count": (
            len(
                raw_normalized
            )
        ),

        "char_count": (
            char_count
        ),

        "alpha_ratio": (
            round(
                alpha_ratio,
                4,
            )
        ),

        "html_detected": (
            raw_html_detected
        ),

        "error_page": (
            raw_error_page
            or final_error_page
        ),

        "transformation": (
            transformation
        ),

        "reasons": (
            reasons
        ),

        "passed": (
            passed
        ),
    }

    return (
        passed,
        prepared_text,
        stats,
    )


# ============================================================
# Metadata Loading
# ============================================================

def _load_axis_candidates(
    topic_axis: str,
) -> list[dict]:

    path = (
        NTRS_CACHE_ROOT
        / topic_axis
        / "_merged_candidates.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"NTRS merged candidates not found: "
            f"{path}"
        )

    payload = _load_json(
        path
    )

    documents = payload.get(
        "documents",
        [],
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

    for document in documents:

        candidate_id = str(
            document.get(
                "source_id",
                "",
            )
        )

        if (
            candidate_id
            == source_id
        ):

            return document

    raise LookupError(
        f"NTRS candidate not found: "
        f"{topic_axis} / {source_id}"
    )


# ============================================================
# URL Helpers
# ============================================================

def _url_extension(
    url: str | None,
) -> str:

    if not url:
        return ""

    parsed = urlparse(
        url
    )

    return (
        Path(
            parsed.path
        )
        .suffix
        .lower()
    )


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


def _is_docx_url(
    url: str | None,
) -> bool:

    if not url:
        return False

    return (
        _url_extension(
            url
        )
        == ".docx"
    )


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
# HTTP Download
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

    response.raise_for_status()

    return response


# ============================================================
# Response Decode
# ============================================================

def _response_to_text(
    response: httpx.Response,
) -> str:
    """
    NASA TXT response decoding을 조금 더 안전하게 처리.
    """

    content = (
        response.content
    )

    if not content:
        return ""

    encoding_candidates = []

    if response.encoding:

        encoding_candidates.append(
            response.encoding
        )

    encoding_candidates.extend(
        [
            "utf-8-sig",
            "utf-8",
            "cp1252",
            "latin-1",
        ]
    )

    tried = set()

    for encoding in (
        encoding_candidates
    ):

        if (
            not encoding
            or encoding in tried
        ):

            continue

        tried.add(
            encoding
        )

        try:

            return content.decode(
                encoding
            )

        except (
            UnicodeDecodeError,
            LookupError,
        ):

            continue

    return content.decode(
        "utf-8",
        errors="replace",
    )


# ============================================================
# NASA Converted TXT Resolver
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

    try:

        raw_text = (
            _response_to_text(
                response
            )
        )

    except Exception as exc:

        print(
            f"[FAIL] TXT decode: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    (
        passed,
        prepared_text,
        quality,
    ) = _evaluate_text(
        raw_text
    )

    print(
        f"[TXT] chars="
        f"{quality['char_count']} "
        f"alpha="
        f"{quality['alpha_ratio']} "
        f"html="
        f"{quality['html_detected']} "
        f"transform="
        f"{quality['transformation']} "
        f"pass="
        f"{quality['passed']}"
    )

    if quality[
        "reasons"
    ]:

        print(
            "[TXT] reasons="
            f"{quality['reasons']}"
        )

    # --------------------------------------------------------
    # 실패 TXT도 디버깅용으로 저장
    # --------------------------------------------------------

    if not passed:

        rejected_path = (
            paper_dir
            / "rejected_fulltext.txt"
        )

        rejected_path.write_text(
            _normalize_text(
                raw_text
            ),
            encoding="utf-8",
            errors="replace",
        )

        print(
            f"[FAIL] TXT quality "
            f"check failed."
        )

        print(
            f"[DEBUG] Rejected TXT saved: "
            f"{rejected_path}"
        )

        return None

    output_path = (
        paper_dir
        / "paper.txt"
    )

    output_path.write_text(
        prepared_text,
        encoding="utf-8",
    )

    return {
        "selected_format": (
            "txt"
        ),

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

        "quality": (
            quality
        ),
    }


# ============================================================
# Text-native Original
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
        f"[TRY] ORIGINAL-TEXT : "
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
            f"[FAIL] ORIGINAL-TEXT HTTP: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    raw_text = (
        _response_to_text(
            response
        )
    )

    (
        passed,
        prepared_text,
        quality,
    ) = _evaluate_text(
        raw_text
    )

    print(
        f"[ORIGINAL-TEXT] "
        f"chars={quality['char_count']} "
        f"alpha={quality['alpha_ratio']} "
        f"html={quality['html_detected']} "
        f"pass={quality['passed']}"
    )

    if not passed:

        print(
            "[FAIL] ORIGINAL-TEXT "
            f"reasons={quality['reasons']}"
        )

        return None

    output_path = (
        paper_dir
        / "paper.txt"
    )

    output_path.write_text(
        prepared_text,
        encoding="utf-8",
    )

    return {
        "selected_format": (
            "original_text"
        ),

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

        "quality": (
            quality
        ),
    }


# ============================================================
# DOCX Extraction
# ============================================================

def _extract_docx_text(
    content: bytes,
) -> str:
    """
    python-docx 없이 표준 라이브러리만 사용.

    DOCX는 ZIP 컨테이너이므로
    word/document.xml의 paragraph text를 추출한다.
    """

    buffer = io.BytesIO(
        content
    )

    if not zipfile.is_zipfile(
        buffer
    ):

        raise ValueError(
            "Downloaded original is not "
            "a valid DOCX/ZIP file."
        )

    buffer.seek(
        0
    )

    with zipfile.ZipFile(
        buffer
    ) as archive:

        names = set(
            archive.namelist()
        )

        required = (
            "word/document.xml"
        )

        if required not in names:

            raise ValueError(
                "DOCX does not contain "
                "word/document.xml."
            )

        document_xml = (
            archive.read(
                required
            )
        )

    root = (
        ElementTree.fromstring(
            document_xml
        )
    )

    namespace = {
        "w": (
            "http://schemas.openxmlformats.org/"
            "wordprocessingml/2006/main"
        )
    }

    paragraphs = []

    for paragraph in root.findall(
        ".//w:p",
        namespace,
    ):

        pieces = []

        for node in paragraph.iter():

            tag = (
                node.tag
                .split(
                    "}"
                )[-1]
            )

            if (
                tag == "t"
                and node.text
            ):

                pieces.append(
                    node.text
                )

            elif tag == "tab":

                pieces.append(
                    "\t"
                )

            elif tag in {
                "br",
                "cr",
            }:

                pieces.append(
                    "\n"
                )

        paragraph_text = (
            "".join(
                pieces
            )
            .strip()
        )

        if paragraph_text:

            paragraphs.append(
                paragraph_text
            )

    text = "\n\n".join(
        paragraphs
    )

    return _normalize_text(
        text
    )


# ============================================================
# DOCX Resolver
# ============================================================

def _try_docx(
    url: str | None,
    paper_dir: Path,
) -> dict | None:

    if not url:
        return None

    print(
        f"[TRY] DOCX     : "
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
            f"[FAIL] DOCX HTTP: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    content = (
        response.content
    )

    if (
        len(content)
        < MIN_DOCX_BYTES
    ):

        print(
            "[FAIL] DOCX file is "
            "suspiciously small."
        )

        return None

    try:

        extracted_text = (
            _extract_docx_text(
                content
            )
        )

    except Exception as exc:

        print(
            f"[FAIL] DOCX parse: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None

    (
        passed,
        prepared_text,
        quality,
    ) = _evaluate_text(
        extracted_text
    )

    print(
        f"[DOCX] chars="
        f"{quality['char_count']} "
        f"alpha="
        f"{quality['alpha_ratio']} "
        f"pass="
        f"{quality['passed']}"
    )

    if not passed:

        print(
            "[FAIL] DOCX text quality "
            f"reasons={quality['reasons']}"
        )

        return None

    # --------------------------------------------------------
    # 원본 DOCX도 provenance를 위해 저장
    # --------------------------------------------------------

    docx_path = (
        paper_dir
        / "paper.docx"
    )

    docx_path.write_bytes(
        content
    )

    # --------------------------------------------------------
    # 이후 parser/cleaner가 공통적으로 읽기 쉽도록
    # 추출 텍스트는 paper.txt
    # --------------------------------------------------------

    text_path = (
        paper_dir
        / "paper.txt"
    )

    text_path.write_text(
        prepared_text,
        encoding="utf-8",
    )

    return {
        "selected_format": (
            "docx_text"
        ),

        "source_url": (
            url
        ),

        "local_file": (
            str(
                text_path
            )
        ),

        "original_file": (
            str(
                docx_path
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

    if not content.startswith(
        b"%PDF"
    ):

        print(
            "[FAIL] Downloaded file "
            "is not a valid PDF."
        )

        return None

    if (
        len(content)
        < MIN_PDF_BYTES
    ):

        print(
            "[FAIL] PDF file is "
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
        "selected_format": (
            "pdf"
        ),

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
# Resolution Save Helper
# ============================================================

def _save_success_resolution(
    *,
    paper_dir: Path,
    topic_axis: str,
    source_id: str,
    title: str,
    result: dict,
    attempts: list[dict],
) -> dict:

    resolution = {
        "source": "ntrs",

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
# Resolve One Document
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

    title = (
        metadata.get(
            "title",
            "",
        )
    )

    print(
        f"[TITLE] {title}"
    )

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
    # Metadata Snapshot
    # ========================================================

    _save_json(
        paper_dir
        / "metadata.json",
        metadata,
    )

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
    # 1. NASA Converted TXT
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
                "format": "txt",

                "url": (
                    fulltext_url
                ),

                "success": (
                    result
                    is not None
                ),
            }
        )

        if result is not None:

            resolution = (
                _save_success_resolution(
                    paper_dir=paper_dir,
                    topic_axis=topic_axis,
                    source_id=source_id,
                    title=title,
                    result=result,
                    attempts=attempts,
                )
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
                    result
                    is not None
                ),
            }
        )

        if result is not None:

            resolution = (
                _save_success_resolution(
                    paper_dir=paper_dir,
                    topic_axis=topic_axis,
                    source_id=source_id,
                    title=title,
                    result=result,
                    attempts=attempts,
                )
            )

            print(
                "[SUCCESS] "
                "text-native original selected."
            )

            return resolution

    # ========================================================
    # 3. DOCX Original
    # ========================================================

    if (
        original_url
        and _is_docx_url(
            original_url
        )
    ):

        result = (
            _try_docx(
                original_url,
                paper_dir,
            )
        )

        attempts.append(
            {
                "format": (
                    "docx_text"
                ),

                "url": (
                    original_url
                ),

                "success": (
                    result
                    is not None
                ),
            }
        )

        if result is not None:

            resolution = (
                _save_success_resolution(
                    paper_dir=paper_dir,
                    topic_axis=topic_axis,
                    source_id=source_id,
                    title=title,
                    result=result,
                    attempts=attempts,
                )
            )

            print(
                "[SUCCESS] "
                "DOCX extracted text selected."
            )

            return resolution

    # ========================================================
    # 4. Explicit PDF URL
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
                "format": "pdf",

                "url": (
                    pdf_url
                ),

                "success": (
                    result
                    is not None
                ),
            }
        )

        if result is not None:

            resolution = (
                _save_success_resolution(
                    paper_dir=paper_dir,
                    topic_axis=topic_axis,
                    source_id=source_id,
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
    # 5. Original URL itself may be PDF
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
                    result
                    is not None
                ),
            }
        )

        if result is not None:

            result[
                "selected_format"
            ] = "pdf"

            resolution = (
                _save_success_resolution(
                    paper_dir=paper_dir,
                    topic_axis=topic_axis,
                    source_id=source_id,
                    title=title,
                    result=result,
                    attempts=attempts,
                )
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
        "source": "ntrs",

        "source_id": (
            source_id
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
        "TEAM B - NASA NTRS Content Resolver v2"
    )

    print("=" * 70)

    print(
        "Priority:"
    )

    print(
        "1. NASA converted TXT"
    )

    print(
        "2. text-native original"
    )

    print(
        "3. DOCX original -> text"
    )

    print(
        "4. PDF fallback"
    )

    print()

    total = 0
    success = 0
    failed = 0

    txt_count = 0
    original_text_count = 0
    docx_text_count = 0
    pdf_count = 0

    results = []

    # ========================================================
    # Resolve 15 Selected Documents
    # ========================================================

    for (
        topic_axis,
        source_ids,
    ) in SELECTED_DOCUMENTS.items():

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

                if selected_format:

                    success += 1

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
                        == "docx_text"
                    ):

                        docx_text_count += 1

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
                        "source": "ntrs",

                        "topic_axis": (
                            topic_axis
                        ),

                        "source_id": (
                            source_id
                        ),

                        "success": False,

                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

            time.sleep(
                settings.request_delay
            )

    # ========================================================
    # Report
    # ========================================================

    report = {
        "total": (
            total
        ),

        "success": (
            success
        ),

        "failed": (
            failed
        ),

        "formats": {
            "txt": (
                txt_count
            ),

            "original_text": (
                original_text_count
            ),

            "docx_text": (
                docx_text_count
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
        "NTRS CONTENT RESOLUTION COMPLETED"
    )

    print("=" * 70)

    print(
        f"Total         : "
        f"{total}"
    )

    print(
        f"Success       : "
        f"{success}"
    )

    print(
        f"Failed        : "
        f"{failed}"
    )

    print(
        f"TXT           : "
        f"{txt_count}"
    )

    print(
        f"Original text : "
        f"{original_text_count}"
    )

    print(
        f"DOCX -> text  : "
        f"{docx_text_count}"
    )

    print(
        f"PDF fallback  : "
        f"{pdf_count}"
    )

    print(
        f"Report        : "
        f"{REPORT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()