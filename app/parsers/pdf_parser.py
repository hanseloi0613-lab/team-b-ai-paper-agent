import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - arXiv PDF Parser
# ============================================================
#
# 목적:
#
# Resolver가 HTML / TeX 확보에 실패하여
# PDF fallback으로 저장한 arXiv 논문을 파싱한다.
#
# 입력:
#
# data/tmp/arxiv_resolved/
#     <topic_axis>/
#         <source_id>/
#             paper.pdf
#             metadata.json
#             resolution.json
#
# 출력:
#
#             raw_content.txt
#             parsed_document.json
#
#
# parsed_document.json 공통 구조:
#
# {
#   "title": "...",
#   "abstract": "...",
#   "sections": [
#       {
#           "heading": "...",
#           "level": 1,
#           "blocks": [
#               {
#                   "type": "paragraph",
#                   "text": "..."
#               }
#           ]
#       }
#   ],
#   "stats": {...}
# }
#
# HTML parser와 동일한 canonical schema를 사용한다.
# ============================================================


# ============================================================
# Paths
# ============================================================

RESOLVED_ROOT = (
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
    / "arxiv_pdf_parsing_report.json"
)


# ============================================================
# Minimum Validation
# ============================================================

MIN_EXTRACTED_CHARS = 500


# ============================================================
# JSON Helpers
# ============================================================

def _load_json(
    path: Path,
) -> dict:

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
    ):
        return {}


# ============================================================
# General Text Helpers
# ============================================================

def _normalize_space(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _clean_line(
    line: str,
) -> str:
    """
    PDF extraction 과정에서 자주 나오는
    특수문자 / ligature / 공백 정리.
    """

    line = (
        line
        .replace(
            "\u00ad",
            "",
        )
        .replace(
            "\u00a0",
            " ",
        )
        .replace(
            "\ufb01",
            "fi",
        )
        .replace(
            "\ufb02",
            "fl",
        )
        .replace(
            "\ufb00",
            "ff",
        )
        .replace(
            "\ufb03",
            "ffi",
        )
        .replace(
            "\ufb04",
            "ffl",
        )
    )

    return _normalize_space(
        line
    )


# ============================================================
# Metadata
# ============================================================

def _load_metadata(
    paper_dir: Path,
) -> dict:

    return _load_json(
        paper_dir
        / "metadata.json"
    )


# ============================================================
# PDF Extraction
# ============================================================

def _extract_pages(
    pdf_path: Path,
) -> list[str]:
    """
    PDF 각 페이지의 텍스트를 추출한다.

    OCR은 사용하지 않는다.
    PDF 내부 text layer가 존재한다는 전제.
    """

    reader = PdfReader(
        str(
            pdf_path
        )
    )

    pages: list[str] = []

    for page in reader.pages:

        try:
            text = (
                page.extract_text()
                or ""
            )

        except Exception:
            text = ""

        # ----------------------------------------------------
        # line-end hyphenation 복구
        #
        # naviga-
        # tion
        #
        # -> navigation
        # ----------------------------------------------------

        text = re.sub(
            r"(?<=\w)-\s*\n\s*(?=\w)",
            "",
            text,
        )

        pages.append(
            text
        )

    return pages


# ============================================================
# Repeated Headers / Footers
# ============================================================

def _detect_repeated_lines(
    pages: list[str],
) -> set[str]:
    """
    여러 페이지에 반복해서 등장하는 짧은 line을
    header/footer 후보로 본다.

    긴 본문 문장은 반복 제거 대상에서 제외한다.
    """

    counter: Counter[str] = Counter()

    for page_text in pages:

        unique_page_lines: set[str] = set()

        for raw_line in (
            page_text.splitlines()
        ):

            line = _clean_line(
                raw_line
            )

            if not line:
                continue

            # 긴 문장은 header/footer로 취급하지 않는다.
            if len(line) > 100:
                continue

            unique_page_lines.add(
                line
            )

        for line in unique_page_lines:

            counter[
                line
            ] += 1

    if not pages:
        return set()

    # 최소 3페이지 또는 전체 페이지의 절반 이상
    threshold = max(
        3,
        len(pages) // 2,
    )

    repeated = {
        line
        for line, count
        in counter.items()
        if count >= threshold
    }

    return repeated


# ============================================================
# Heading Detection
# ============================================================

KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "motivation",
    "related work",
    "related works",
    "literature review",
    "method",
    "methods",
    "methodology",
    "approach",
    "proposed approach",
    "system overview",
    "architecture",
    "model",
    "problem formulation",
    "problem statement",
    "experimental setup",
    "experiment setup",
    "experiments",
    "experiment",
    "evaluation",
    "results",
    "results and discussion",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "summary",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "references",
    "bibliography",
}


def _normalize_heading_name(
    text: str,
) -> str:

    normalized = re.sub(
        r"^\s*"
        r"(?:"
        r"\d+(?:\.\d+)*"
        r"|[IVXLC]+"
        r")"
        r"[\.\s:-]*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return (
        normalized
        .strip()
        .lower()
        .rstrip(
            ":."
        )
    )


def _looks_like_heading(
    line: str,
) -> bool:

    text = line.strip()

    if not text:
        return False

    # 너무 긴 것은 제목으로 판단하지 않는다.
    if len(text) > 140:
        return False

    word_count = len(
        text.split()
    )

    if word_count > 18:
        return False

    normalized = (
        _normalize_heading_name(
            text
        )
    )

    # --------------------------------------------------------
    # Known academic headings
    # --------------------------------------------------------

    if normalized in KNOWN_HEADINGS:
        return True

    # --------------------------------------------------------
    # 1 Introduction
    # 2. Method
    # 2.3 Autonomous Navigation
    # --------------------------------------------------------

    if re.match(
        r"^\d+(?:\.\d+)*"
        r"\s*[.\-:]?\s+"
        r"[A-Z]",
        text,
    ):
        return True

    # --------------------------------------------------------
    # I. INTRODUCTION
    # II. METHODS
    # --------------------------------------------------------

    if re.match(
        r"^[IVXLC]+"
        r"\.\s+"
        r"[A-Z]",
        text,
    ):
        return True

    # --------------------------------------------------------
    # ALL CAPS HEADING
    #
    # AUTONOMOUS NAVIGATION
    # EXPERIMENTAL RESULTS
    # --------------------------------------------------------

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if (
        letters
        and word_count <= 12
        and len(letters) >= 4
        and all(
            char.isupper()
            for char in letters
        )
    ):
        return True

    return False


def _heading_level(
    line: str,
) -> int:
    """
    1
    1.2
    1.2.3

    를 기준으로 section level 추정.
    """

    stripped = (
        line.strip()
    )

    match = re.match(
        r"^(\d+(?:\.\d+)*)",
        stripped,
    )

    if match:

        return min(
            3,
            match.group(
                1
            ).count(
                "."
            ) + 1,
        )

    return 1


# ============================================================
# Block Detection
# ============================================================

def _is_figure_caption(
    text: str,
) -> bool:

    return bool(
        re.match(
            r"^(?:fig\.?|figure)\s*\d+",
            text,
            flags=re.IGNORECASE,
        )
    )


def _is_table_caption(
    text: str,
) -> bool:

    return bool(
        re.match(
            r"^table\s*[IVXLC\d]+",
            text,
            flags=re.IGNORECASE,
        )
    )


# ============================================================
# Pages -> Sections
# ============================================================

def _pages_to_sections(
    pages: list[str],
) -> list[dict]:
    """
    PDF 추출 text를 HTML parser와 같은:

        section
            heading
            level
            blocks

    구조로 변환한다.
    """

    repeated_lines = (
        _detect_repeated_lines(
            pages
        )
    )

    sections: list[dict] = []

    current_section = {
        "heading": "Document Body",
        "level": 1,
        "blocks": [],
    }

    paragraph_buffer: list[str] = []

    # ========================================================
    # Flush Paragraph
    # ========================================================

    def flush_paragraph() -> None:

        nonlocal paragraph_buffer

        if not paragraph_buffer:
            return

        paragraph = _normalize_space(
            " ".join(
                paragraph_buffer
            )
        )

        paragraph_buffer = []

        if not paragraph:
            return

        if _is_figure_caption(
            paragraph
        ):

            block_type = (
                "figure_caption"
            )

        elif _is_table_caption(
            paragraph
        ):

            block_type = (
                "table_caption"
            )

        else:

            block_type = (
                "paragraph"
            )

        current_section[
            "blocks"
        ].append(
            {
                "type": (
                    block_type
                ),
                "text": (
                    paragraph
                ),
            }
        )

    # ========================================================
    # Page Processing
    # ========================================================

    for page_text in pages:

        raw_lines = (
            page_text.splitlines()
        )

        for raw_line in raw_lines:

            line = (
                _clean_line(
                    raw_line
                )
            )

            # ------------------------------------------------
            # Blank line
            # ------------------------------------------------

            if not line:

                flush_paragraph()
                continue

            # ------------------------------------------------
            # Repeated Header / Footer
            # ------------------------------------------------

            if line in repeated_lines:
                continue

            # ------------------------------------------------
            # Page number
            #
            # 1
            # Page 1
            # ------------------------------------------------

            if re.fullmatch(
                r"(?:page\s*)?\d+",
                line,
                flags=re.IGNORECASE,
            ):
                continue

            # ------------------------------------------------
            # Common arXiv footer
            # ------------------------------------------------

            if re.fullmatch(
                r"arXiv:\S+",
                line,
                flags=re.IGNORECASE,
            ):
                continue

            # ------------------------------------------------
            # Heading
            # ------------------------------------------------

            if _looks_like_heading(
                line
            ):

                flush_paragraph()

                if current_section[
                    "blocks"
                ]:

                    sections.append(
                        current_section
                    )

                current_section = {
                    "heading": (
                        line
                    ),
                    "level": (
                        _heading_level(
                            line
                        )
                    ),
                    "blocks": [],
                }

                continue

            # ------------------------------------------------
            # Normal Body
            # ------------------------------------------------

            paragraph_buffer.append(
                line
            )

            # 지나치게 긴 buffer 방지.
            #
            # PDF extract에서 paragraph boundary가
            # 사라지는 경우를 대비한다.
            if (
                line.endswith(
                    (
                        ".",
                        "!",
                        "?",
                    )
                )
                and len(
                    " ".join(
                        paragraph_buffer
                    )
                ) >= 500
            ):

                flush_paragraph()

        # 페이지 경계에서 paragraph 종료
        flush_paragraph()

    # ========================================================
    # Final Flush
    # ========================================================

    flush_paragraph()

    if current_section[
        "blocks"
    ]:

        sections.append(
            current_section
        )

    return sections


# ============================================================
# Raw Content Rendering
# ============================================================

def _render_raw_content(
    *,
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:
    """
    title + abstract + structured section을
    사람이 읽을 수 있는 plain text로 변환.
    """

    parts: list[str] = []

    if title:

        parts.append(
            title
        )

    if abstract:

        parts.append(
            "Abstract"
        )

        parts.append(
            abstract
        )

    for section in sections:

        heading = str(
            section.get(
                "heading",
                "",
            )
            or ""
        ).strip()

        if heading:

            parts.append(
                heading
            )

        for block in section.get(
            "blocks",
            [],
        ):

            text = str(
                block.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if text:

                parts.append(
                    text
                )

    return "\n\n".join(
        parts
    ).strip()


# ============================================================
# Statistics
# ============================================================

def _build_stats(
    raw_content: str,
    sections: list[dict],
    page_count: int,
) -> dict:

    paragraph_count = 0
    equation_count = 0
    figure_caption_count = 0
    table_caption_count = 0

    for section in sections:

        for block in section.get(
            "blocks",
            [],
        ):

            block_type = (
                block.get(
                    "type"
                )
            )

            if (
                block_type
                == "paragraph"
            ):

                paragraph_count += 1

            elif (
                block_type
                == "equation"
            ):

                equation_count += 1

            elif (
                block_type
                == "figure_caption"
            ):

                figure_caption_count += 1

            elif (
                block_type
                == "table_caption"
            ):

                table_caption_count += 1

    return {
        "char_count": len(
            raw_content
        ),

        "word_count": len(
            raw_content.split()
        ),

        "section_count": len(
            sections
        ),

        "paragraph_count": (
            paragraph_count
        ),

        "equation_count": (
            equation_count
        ),

        "figure_caption_count": (
            figure_caption_count
        ),

        "table_caption_count": (
            table_caption_count
        ),

        "page_count": (
            page_count
        ),
    }


# ============================================================
# Parse One PDF
# ============================================================

def parse_pdf_file(
    pdf_path: Path,
) -> dict:
    """
    PDF 한 편을 canonical parsed-document 구조로 변환.
    """

    paper_dir = (
        pdf_path.parent
    )

    metadata = (
        _load_metadata(
            paper_dir
        )
    )

    # ========================================================
    # Title
    # ========================================================

    title = _normalize_space(
        str(
            metadata.get(
                "title",
                "",
            )
            or ""
        )
    )

    # ========================================================
    # Abstract
    #
    # arXiv metadata의 abstract를 우선 사용한다.
    # PDF 본문에서 abstract를 다시 추정할 필요가 없음.
    # ========================================================

    abstract = _normalize_space(
        str(
            metadata.get(
                "abstract",
                "",
            )
            or ""
        )
    )

    # ========================================================
    # Pages
    # ========================================================

    pages = (
        _extract_pages(
            pdf_path
        )
    )

    if not pages:

        raise RuntimeError(
            "PDF contains no pages."
        )

    extracted_char_count = sum(
        len(
            page.strip()
        )
        for page in pages
    )

    if (
        extracted_char_count
        < MIN_EXTRACTED_CHARS
    ):

        raise RuntimeError(
            "PDF text extraction produced "
            f"too little text: "
            f"{extracted_char_count} chars"
        )

    # ========================================================
    # Sections
    # ========================================================

    sections = (
        _pages_to_sections(
            pages
        )
    )

    if not sections:

        raise RuntimeError(
            "No sections could be constructed "
            "from extracted PDF text."
        )

    # ========================================================
    # Raw Content
    # ========================================================

    raw_content = (
        _render_raw_content(
            title=title,
            abstract=abstract,
            sections=sections,
        )
    )

    if not raw_content.strip():

        raise RuntimeError(
            "Parsed PDF content is empty."
        )

    # ========================================================
    # Stats
    # ========================================================

    stats = (
        _build_stats(
            raw_content,
            sections,
            len(pages),
        )
    )

    return {
        "title": (
            title
        ),

        "abstract": (
            abstract
        ),

        "sections": (
            sections
        ),

        "raw_content": (
            raw_content
        ),

        "stats": (
            stats
        ),
    }


# ============================================================
# Save Parsed Document
# ============================================================

def _save_parsed_document(
    paper_dir: Path,
    parsed: dict,
) -> None:

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    parsed_path = (
        paper_dir
        / "parsed_document.json"
    )

    # ========================================================
    # raw_content.txt
    # ========================================================

    raw_path.write_text(
        parsed[
            "raw_content"
        ],
        encoding="utf-8",
    )

    # ========================================================
    # parsed_document.json
    #
    # HTML parser와 동일한 canonical 구조.
    # ========================================================

    payload = {
        "title": (
            parsed[
                "title"
            ]
        ),

        "abstract": (
            parsed[
                "abstract"
            ]
        ),

        "sections": (
            parsed[
                "sections"
            ]
        ),

        "stats": (
            parsed[
                "stats"
            ]
        ),
    }

    parsed_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Find All PDF Papers
# ============================================================

def _find_pdf_files() -> list[Path]:
    """
    resolver가 PDF fallback으로 저장한
    모든 paper.pdf 탐색.
    """

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            f"Resolved root not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/paper.pdf"
        )
    )


# ============================================================
# Report
# ============================================================

def _save_report(
    results: list[dict],
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = sum(
        1
        for item in results
        if item[
            "status"
        ] == "success"
    )

    failed_count = (
        len(results)
        - success_count
    )

    payload = {
        "total": (
            len(results)
        ),

        "success": (
            success_count
        ),

        "failed": (
            failed_count
        ),

        "documents": (
            results
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv PDF Parser"
    )

    print("=" * 70)

    print(
        f"Input root : "
        f"{RESOLVED_ROOT}"
    )

    pdf_files = (
        _find_pdf_files()
    )

    print(
        f"PDF files  : "
        f"{len(pdf_files)}"
    )

    if not pdf_files:

        print()
        print(
            "[INFO] No PDF files found."
        )

        return

    results: list[dict] = []

    total_words = 0
    total_chars = 0

    # ========================================================
    # Parse
    # ========================================================

    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        paper_dir = (
            pdf_path.parent
        )

        topic_axis = (
            paper_dir.parent.name
        )

        source_id = (
            paper_dir.name
        )

        print()
        print("-" * 70)

        print(
            f"[{index}/"
            f"{len(pdf_files)}]"
        )

        print(
            f"[AXIS] "
            f"{topic_axis}"
        )

        print(
            f"[ID]   "
            f"{source_id}"
        )

        print(
            f"[FILE] "
            f"{pdf_path}"
        )

        print("-" * 70)

        try:

            parsed = (
                parse_pdf_file(
                    pdf_path
                )
            )

            _save_parsed_document(
                paper_dir,
                parsed,
            )

            stats = (
                parsed[
                    "stats"
                ]
            )

            total_words += int(
                stats[
                    "word_count"
                ]
            )

            total_chars += int(
                stats[
                    "char_count"
                ]
            )

            print(
                f"[OK] Title      : "
                f"{parsed['title']}"
            )

            print(
                f"[OK] Pages      : "
                f"{stats['page_count']}"
            )

            print(
                f"[OK] Sections   : "
                f"{stats['section_count']}"
            )

            print(
                f"[OK] Paragraphs : "
                f"{stats['paragraph_count']}"
            )

            print(
                f"[OK] Figures    : "
                f"{stats['figure_caption_count']}"
            )

            print(
                f"[OK] Tables     : "
                f"{stats['table_caption_count']}"
            )

            print(
                f"[OK] Words      : "
                f"{stats['word_count']}"
            )

            print(
                f"[OK] Chars      : "
                f"{stats['char_count']}"
            )

            print(
                f"[SAVE] "
                f"{paper_dir / 'raw_content.txt'}"
            )

            print(
                f"[SAVE] "
                f"{paper_dir / 'parsed_document.json'}"
            )

            results.append(
                {
                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "status": (
                        "success"
                    ),

                    "stats": (
                        stats
                    ),
                }
            )

        except Exception as exc:

            print(
                f"[FAIL] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            results.append(
                {
                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "status": (
                        "failed"
                    ),

                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    # ========================================================
    # Report
    # ========================================================

    _save_report(
        results
    )

    success_count = sum(
        1
        for item in results
        if item[
            "status"
        ] == "success"
    )

    failed_count = (
        len(results)
        - success_count
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "PDF parsing completed."
    )

    print("=" * 70)

    print(
        f"Total       : "
        f"{len(results)}"
    )

    print(
        f"Success     : "
        f"{success_count}"
    )

    print(
        f"Failed      : "
        f"{failed_count}"
    )

    print(
        f"Total words : "
        f"{total_words}"
    )

    print(
        f"Total chars : "
        f"{total_chars}"
    )

    print(
        f"Report      : "
        f"{REPORT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()