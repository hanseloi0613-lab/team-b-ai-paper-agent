import json
import re
import unicodedata
from pathlib import Path

from app.config import PROJECT_ROOT


# ============================================================
# Paths
# ============================================================

RESOLVED_ROOT = (
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
    / "ntrs_txt_parsing_report.json"
)


# ============================================================
# Heading Vocabulary
# ============================================================
#
# NASA 기술보고서 / conference paper에서 자주 나오는
# section heading.
#
# 이것만 사용하는 것은 아니고,
# 아래의 numbered heading / ALL CAPS heading도 인식한다.
#
# ============================================================

KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "motivation",
    "overview",
    "system overview",
    "architecture",
    "system architecture",
    "method",
    "methods",
    "methodology",
    "approach",
    "proposed approach",
    "algorithm",
    "algorithms",
    "implementation",
    "experimental setup",
    "experiment",
    "experiments",
    "evaluation",
    "analysis",
    "results",
    "discussion",
    "results and discussion",
    "conclusion",
    "conclusions",
    "concluding remarks",
    "summary",
    "future work",
    "references",
    "reference",
    "bibliography",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "appendix",
    "appendices",
}


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
        UnicodeDecodeError,
    ):

        return {}


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
# Basic Text Normalization
# ============================================================

def _normalize_text(
    text: str,
) -> str:
    """
    parser 단계에서는 과도하게 clean하지 않는다.

    목적:
    - encoding/control character 정리
    - 줄바꿈 통일
    - NASA TXT의 기본 구조 보존

    bibliography 제거 같은 실제 cleaning은
    cleaner.py 단계에서 수행한다.
    """

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    text = text.replace(
        "\x00",
        "",
    )

    text = text.replace(
        "\ufeff",
        "",
    )

    text = text.replace(
        "\xa0",
        " ",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # form feed = PDF/page boundary 성격
    text = text.replace(
        "\f",
        "\n\n",
    )

    # tab -> space
    text = text.replace(
        "\t",
        " ",
    )

    # line 안에서만 과도한 공백 제거
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text,
    )

    # 지나친 blank line 축소
    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    return text.strip()


# ============================================================
# Metadata
# ============================================================

def _load_document_metadata(
    paper_dir: Path,
) -> tuple[
    dict,
    dict,
]:
    """
    resolver가 생성한:

        metadata.json
        resolution.json

    을 읽는다.

    title / abstract는 가능하면
    NASA metadata를 신뢰한다.
    """

    metadata = (
        _load_json(
            paper_dir
            / "metadata.json"
        )
    )

    resolution = (
        _load_json(
            paper_dir
            / "resolution.json"
        )
    )

    return (
        metadata,
        resolution,
    )


def _extract_title(
    metadata: dict,
    resolution: dict,
    paper_dir: Path,
) -> str:

    title = (
        metadata.get(
            "title"
        )
        or resolution.get(
            "title"
        )
        or ""
    )

    title = " ".join(
        str(
            title
        ).split()
    )

    if title:

        return title

    # 최후 fallback
    return paper_dir.name


def _extract_abstract(
    metadata: dict,
) -> str:

    abstract = (
        metadata.get(
            "abstract"
        )
        or ""
    )

    abstract = str(
        abstract
    )

    abstract = re.sub(
        r"\s+",
        " ",
        abstract,
    )

    return abstract.strip()


# ============================================================
# Line Helpers
# ============================================================

def _clean_line(
    line: str,
) -> str:

    line = line.strip()

    line = re.sub(
        r"[ \t]+",
        " ",
        line,
    )

    return line.strip()


def _normalized_compare(
    text: str,
) -> str:
    """
    title 중복 등을 비교할 때만 사용하는
    느슨한 normalized representation.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


# ============================================================
# Noise Line Detection
# ============================================================

def _is_page_number(
    line: str,
) -> bool:
    """
    단독 page number:
        1
        12
        123
    """

    return bool(
        re.fullmatch(
            r"\d{1,4}",
            line.strip(),
        )
    )


def _is_separator_line(
    line: str,
) -> bool:

    stripped = line.strip()

    if len(
        stripped
    ) < 3:

        return False

    return bool(
        re.fullmatch(
            r"[-_=*•·.]{3,}",
            stripped,
        )
    )


def _is_probable_noise(
    line: str,
) -> bool:
    """
    아주 명확한 layout noise만 제거한다.

    내용일 가능성이 있는 것은 버리지 않는다.
    """

    stripped = line.strip()

    if not stripped:

        return False

    if _is_separator_line(
        stripped
    ):

        return True

    # 단독 page number
    if _is_page_number(
        stripped
    ):

        return True

    # PDF converter의 흔한 page marker
    if re.fullmatch(
        r"page\s+\d+\s*(?:of\s+\d+)?",
        stripped,
        flags=re.IGNORECASE,
    ):

        return True

    return False


# ============================================================
# Heading Detection
# ============================================================

def _strip_heading_number(
    line: str,
) -> str:
    """
    예:
        1. INTRODUCTION
        2.1 Navigation
        IV. RESULTS
        A. System Architecture

    앞 번호만 제거.
    """

    result = re.sub(
        (
            r"^\s*"
            r"(?:"
            r"\d+(?:\.\d+)*"
            r"|[IVXLC]+"
            r"|[A-Z]"
            r")"
            r"[\.\)]?"
            r"\s+"
        ),
        "",
        line,
        flags=re.IGNORECASE,
    )

    return result.strip()


def _looks_like_sentence(
    line: str,
) -> bool:

    stripped = line.strip()

    if not stripped:

        return False

    # 긴 line + sentence punctuation이면
    # heading으로 보지 않음
    if (
        len(
            stripped
        ) > 80
        and stripped.endswith(
            (
                ".",
                "?",
                "!",
                ";",
            )
        )
    ):

        return True

    return False


def _is_known_heading(
    line: str,
) -> bool:

    normalized = (
        _strip_heading_number(
            line
        )
        .lower()
        .strip(" :.-")
    )

    return (
        normalized
        in KNOWN_HEADINGS
    )


def _is_numbered_heading(
    line: str,
) -> bool:
    """
    일반적인 technical paper section heading.

    예:
        1 Introduction
        1. Introduction
        2.1 System Design
        III. RESULTS
        A. Architecture
    """

    stripped = line.strip()

    if (
        len(
            stripped
        ) < 4
        or len(
            stripped
        ) > 140
    ):

        return False

    pattern = re.compile(
        (
            r"^(?:"
            r"\d+(?:\.\d+)*"
            r"|[IVXLC]+"
            r"|[A-Z]"
            r")"
            r"[\.\)]?"
            r"\s+"
            r".{2,120}$"
        ),
        flags=re.IGNORECASE,
    )

    if not pattern.match(
        stripped
    ):

        return False

    heading_text = (
        _strip_heading_number(
            stripped
        )
    )

    # 실제 문장처럼 너무 길면 제외
    if _looks_like_sentence(
        heading_text
    ):

        return False

    # heading 자체가 최소한 alphabet을 포함
    alpha_count = sum(
        1
        for char in heading_text
        if char.isalpha()
    )

    if alpha_count < 2:

        return False

    return True


def _is_all_caps_heading(
    line: str,
) -> bool:

    stripped = line.strip()

    if (
        len(
            stripped
        ) < 3
        or len(
            stripped
        ) > 120
    ):

        return False

    # email / URL
    if (
        "@" in stripped
        or "http://" in stripped.lower()
        or "https://" in stripped.lower()
    ):

        return False

    letters = [
        char
        for char in stripped
        if char.isalpha()
    ]

    if len(
        letters
    ) < 3:

        return False

    uppercase_ratio = (
        sum(
            1
            for char in letters
            if char.isupper()
        )
        / len(
            letters
        )
    )

    if uppercase_ratio < 0.90:

        return False

    word_count = len(
        re.findall(
            r"[A-Za-z]+",
            stripped,
        )
    )

    if word_count > 14:

        return False

    # 완전한 문장 가능성 낮은 경우만
    if stripped.endswith(
        (
            ".",
            ";",
            ",",
        )
    ):

        return False

    return True


def _is_title_case_heading(
    line: str,
) -> bool:
    """
    번호 없는 Title Case heading을
    매우 보수적으로 검출.

    예:
        System Architecture
        Fault Detection Approach
        Experimental Results
    """

    stripped = line.strip()

    if (
        len(
            stripped
        ) < 5
        or len(
            stripped
        ) > 80
    ):

        return False

    if stripped.endswith(
        (
            ".",
            ",",
            ";",
            ":",
            "?",
            "!",
        )
    ):

        return False

    words = re.findall(
        r"[A-Za-z][A-Za-z0-9/-]*",
        stripped,
    )

    if not (
        2
        <= len(
            words
        )
        <= 8
    ):

        return False

    # lowercase function words는 허용
    allowed_lower = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }

    capitalized = 0

    for word in words:

        if word.lower() in allowed_lower:

            continue

        if (
            word[0].isupper()
            or word.isupper()
        ):

            capitalized += 1

    meaningful = [
        word
        for word in words
        if word.lower()
        not in allowed_lower
    ]

    if not meaningful:

        return False

    ratio = (
        capitalized
        / len(
            meaningful
        )
    )

    if ratio < 0.80:

        return False

    # 전문 section 단어가 하나라도 포함되는 경우에만
    heading_terms = {
        "architecture",
        "approach",
        "analysis",
        "background",
        "conclusion",
        "conclusions",
        "design",
        "discussion",
        "evaluation",
        "experiment",
        "experiments",
        "implementation",
        "introduction",
        "method",
        "methods",
        "methodology",
        "model",
        "navigation",
        "overview",
        "planning",
        "results",
        "system",
        "systems",
        "testing",
        "validation",
    }

    return any(
        word.lower()
        in heading_terms
        for word in words
    )


def _is_heading(
    line: str,
) -> bool:

    stripped = line.strip()

    if not stripped:

        return False

    if _is_known_heading(
        stripped
    ):

        return True

    if _is_numbered_heading(
        stripped
    ):

        return True

    if _is_all_caps_heading(
        stripped
    ):

        return True

    if _is_title_case_heading(
        stripped
    ):

        return True

    return False


def _heading_level(
    line: str,
) -> int:
    """
    완벽한 hierarchy 재구성이 목적이 아니라
    section 구조 보존 목적.

    1        -> level 2
    1.2      -> level 3
    1.2.3    -> level 4
    """

    stripped = line.strip()

    match = re.match(
        r"^(\d+(?:\.\d+)*)",
        stripped,
    )

    if match:

        number = match.group(
            1
        )

        depth = (
            number.count(
                "."
            )
            + 1
        )

        return min(
            6,
            depth
            + 1,
        )

    return 2


# ============================================================
# Block Detection
# ============================================================

def _is_figure_caption(
    text: str,
) -> bool:

    return bool(
        re.match(
            (
                r"^(?:"
                r"fig(?:ure)?\.?"
                r"|table"
                r")"
                r"\s*"
                r"[A-Za-z0-9IVXLC\-]+"
                r"[\s\.:]"
            ),
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def _is_list_item(
    text: str,
) -> bool:

    return bool(
        re.match(
            r"^(?:[-•*]|\([a-z0-9]+\))\s+",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def _strip_list_marker(
    text: str,
) -> str:

    return re.sub(
        r"^(?:[-•*]|\([a-z0-9]+\))\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()


def _looks_like_equation(
    text: str,
) -> bool:
    """
    아주 명확한 standalone equation만 검출.

    일반 문장을 equation으로 오판하지 않도록
    매우 보수적으로 동작.
    """

    stripped = text.strip()

    if (
        len(
            stripped
        ) < 3
        or len(
            stripped
        ) > 250
    ):

        return False

    if len(
        stripped.split()
    ) > 25:

        return False

    math_symbols = sum(
        stripped.count(
            symbol
        )
        for symbol in (
            "=",
            "±",
            "∑",
            "∫",
            "≤",
            "≥",
            "≈",
            "→",
            "λ",
            "μ",
            "σ",
            "Δ",
            "θ",
        )
    )

    if math_symbols == 0:

        return False

    alpha_words = len(
        re.findall(
            r"\b[A-Za-z]{4,}\b",
            stripped,
        )
    )

    return (
        math_symbols >= 1
        and alpha_words <= 5
    )


# ============================================================
# Paragraph Joining
# ============================================================

def _join_paragraph_lines(
    lines: list[str],
) -> str:

    if not lines:

        return ""

    text = " ".join(
        line.strip()
        for line in lines
        if line.strip()
    )

    # line-wrap hyphen:
    #
    # "autono- mous" 같은 변환 잔여물 일부 복원.
    #
    # 이미 space가 들어간 상태이므로
    # alphabet-hyphen-space-alphabet만 처리.
    #
    # 단, 일반적인 "fault- tolerant" 같은 표현까지
    # 무조건 합치지 않도록 앞뒤가 소문자인 경우 중심.
    text = re.sub(
        r"(?<=[a-z])-\s+(?=[a-z])",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Parse Lines -> Blocks
# ============================================================

def _extract_blocks(
    text: str,
    title: str,
) -> list[dict]:
    """
    NASA converted TXT를 block stream으로 변환.

    block:
        heading
        paragraph
        list_item
        figure_caption
        equation
    """

    raw_lines = (
        text.splitlines()
    )

    lines: list[str] = []

    title_key = (
        _normalized_compare(
            title
        )
    )

    title_skipped = False

    for raw_line in raw_lines:

        line = (
            _clean_line(
                raw_line
            )
        )

        # blank line은 paragraph boundary로 필요
        if not line:

            lines.append(
                ""
            )

            continue

        if _is_probable_noise(
            line
        ):

            continue

        # 문서 앞쪽 title 중복 1회 제거
        if (
            not title_skipped
            and title_key
            and _normalized_compare(
                line
            )
            == title_key
        ):

            title_skipped = True

            continue

        lines.append(
            line
        )

    blocks: list[dict] = []

    paragraph_buffer: list[
        str
    ] = []

    def flush_paragraph():

        if not paragraph_buffer:

            return

        paragraph = (
            _join_paragraph_lines(
                paragraph_buffer
            )
        )

        paragraph_buffer.clear()

        if not paragraph:

            return

        # figure caption
        if _is_figure_caption(
            paragraph
        ):

            blocks.append(
                {
                    "type": (
                        "figure_caption"
                    ),

                    "text": (
                        paragraph
                    ),
                }
            )

            return

        # list item
        if _is_list_item(
            paragraph
        ):

            blocks.append(
                {
                    "type": (
                        "list_item"
                    ),

                    "text": (
                        _strip_list_marker(
                            paragraph
                        )
                    ),
                }
            )

            return

        # equation
        if _looks_like_equation(
            paragraph
        ):

            blocks.append(
                {
                    "type": (
                        "equation"
                    ),

                    "text": (
                        paragraph
                    ),
                }
            )

            return

        blocks.append(
            {
                "type": (
                    "paragraph"
                ),

                "text": (
                    paragraph
                ),
            }
        )

    for line in lines:

        # ----------------------------------------------------
        # Blank line
        # ----------------------------------------------------

        if not line:

            flush_paragraph()

            continue

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if _is_heading(
            line
        ):

            flush_paragraph()

            blocks.append(
                {
                    "type": (
                        "heading"
                    ),

                    "level": (
                        _heading_level(
                            line
                        )
                    ),

                    "text": (
                        line
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Standalone Figure/Table Caption
        # ----------------------------------------------------

        if _is_figure_caption(
            line
        ):

            flush_paragraph()

            blocks.append(
                {
                    "type": (
                        "figure_caption"
                    ),

                    "text": (
                        line
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Standalone List Item
        # ----------------------------------------------------

        if _is_list_item(
            line
        ):

            flush_paragraph()

            blocks.append(
                {
                    "type": (
                        "list_item"
                    ),

                    "text": (
                        _strip_list_marker(
                            line
                        )
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Standalone Equation
        # ----------------------------------------------------

        if _looks_like_equation(
            line
        ):

            flush_paragraph()

            blocks.append(
                {
                    "type": (
                        "equation"
                    ),

                    "text": (
                        line
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Normal paragraph continuation
        # ----------------------------------------------------

        paragraph_buffer.append(
            line
        )

    flush_paragraph()

    return blocks


# ============================================================
# Blocks -> Sections
# ============================================================

def _is_abstract_heading(
    heading: str,
) -> bool:

    normalized = (
        _strip_heading_number(
            heading
        )
        .lower()
        .strip(" :.-")
    )

    return (
        normalized
        == "abstract"
    )


def _blocks_to_sections(
    blocks: list[dict],
) -> list[dict]:
    """
    기존 arXiv HTML parser와 동일한
    section schema를 만든다.

    {
        "heading": ...,
        "level": ...,
        "blocks": [...]
    }

    metadata의 abstract를 별도 필드로 저장하므로
    본문에 다시 등장하는 ABSTRACT section은
    중복 방지를 위해 sections에서 제외한다.
    """

    sections: list[dict] = []

    current_section = {
        "heading": (
            "Document Body"
        ),

        "level": 1,

        "blocks": [],
    }

    skip_current = False

    for block in blocks:

        if (
            block.get(
                "type"
            )
            == "heading"
        ):

            # 기존 section 저장
            if (
                current_section[
                    "blocks"
                ]
                and not skip_current
            ):

                sections.append(
                    current_section
                )

            heading = str(
                block.get(
                    "text",
                    "",
                )
            ).strip()

            current_section = {
                "heading": heading,

                "level": (
                    block.get(
                        "level",
                        2,
                    )
                ),

                "blocks": [],
            }

            skip_current = (
                _is_abstract_heading(
                    heading
                )
            )

            continue

        current_section[
            "blocks"
        ].append(
            block
        )

    if (
        current_section[
            "blocks"
        ]
        and not skip_current
    ):

        sections.append(
            current_section
        )

    return sections


# ============================================================
# Duplicate Empty Sections
# ============================================================

def _remove_empty_sections(
    sections: list[dict],
) -> list[dict]:

    result = []

    for section in sections:

        blocks = (
            section.get(
                "blocks",
                [],
            )
        )

        if not blocks:

            continue

        useful_blocks = [
            block
            for block in blocks
            if str(
                block.get(
                    "text",
                    "",
                )
            ).strip()
        ]

        if not useful_blocks:

            continue

        copied = dict(
            section
        )

        copied[
            "blocks"
        ] = useful_blocks

        result.append(
            copied
        )

    return result


# ============================================================
# Raw Content Renderer
# ============================================================

def _render_raw_content(
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:
    """
    기존 arXiv parser와 동일한 내부 표현.

    # TITLE
    ...
    ## ABSTRACT
    ...
    ## INTRODUCTION
    ...
    """

    output: list[str] = []

    # ========================================================
    # Title
    # ========================================================

    if title:

        output.append(
            "# TITLE"
        )

        output.append(
            title
        )

        output.append(
            ""
        )

    # ========================================================
    # Abstract
    # ========================================================

    if abstract:

        output.append(
            "## ABSTRACT"
        )

        output.append(
            abstract
        )

        output.append(
            ""
        )

    # ========================================================
    # Sections
    # ========================================================

    for section in sections:

        heading = str(
            section.get(
                "heading",
                "Document Body",
            )
        )

        level = int(
            section.get(
                "level",
                2,
            )
        )

        markdown_level = max(
            2,
            min(
                level,
                6,
            ),
        )

        output.append(
            (
                "#" * markdown_level
            )
            + " "
            + heading
        )

        output.append(
            ""
        )

        for block in section.get(
            "blocks",
            [],
        ):

            block_type = (
                block.get(
                    "type"
                )
            )

            block_text = str(
                block.get(
                    "text",
                    "",
                )
            ).strip()

            if not block_text:

                continue

            if (
                block_type
                == "list_item"
            ):

                output.append(
                    f"- {block_text}"
                )

            elif (
                block_type
                == "figure_caption"
            ):

                output.append(
                    (
                        "[FIGURE CAPTION] "
                        f"{block_text}"
                    )
                )

            elif (
                block_type
                == "equation"
            ):

                output.append(
                    (
                        "[EQUATION] "
                        f"{block_text}"
                    )
                )

            else:

                output.append(
                    block_text
                )

            output.append(
                ""
            )

    result = "\n".join(
        output
    )

    result = re.sub(
        r"\n{3,}",
        "\n\n",
        result,
    )

    return result.strip()


# ============================================================
# Statistics
# ============================================================

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

    return round(
        alpha_count
        / len(
            text
        ),
        4,
    )


def _build_stats(
    raw_content: str,
    sections: list[dict],
) -> dict:

    paragraph_count = 0
    list_item_count = 0
    equation_count = 0
    figure_caption_count = 0

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
                == "list_item"
            ):

                list_item_count += 1

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

        "list_item_count": (
            list_item_count
        ),

        "equation_count": (
            equation_count
        ),

        "figure_caption_count": (
            figure_caption_count
        ),

        "alpha_ratio": (
            _alpha_ratio(
                raw_content
            )
        ),
    }


# ============================================================
# Parse One NTRS TXT
# ============================================================

def parse_txt_file(
    txt_path: Path,
) -> dict:

    paper_dir = (
        txt_path.parent
    )

    metadata, resolution = (
        _load_document_metadata(
            paper_dir
        )
    )

    title = (
        _extract_title(
            metadata,
            resolution,
            paper_dir,
        )
    )

    abstract = (
        _extract_abstract(
            metadata
        )
    )

    source_text = (
        txt_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    source_text = (
        _normalize_text(
            source_text
        )
    )

    blocks = (
        _extract_blocks(
            text=source_text,
            title=title,
        )
    )

    sections = (
        _blocks_to_sections(
            blocks
        )
    )

    sections = (
        _remove_empty_sections(
            sections
        )
    )

    raw_content = (
        _render_raw_content(
            title=title,
            abstract=abstract,
            sections=sections,
        )
    )

    stats = (
        _build_stats(
            raw_content,
            sections,
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

        "source_info": {
            "source": "ntrs",

            "source_id": (
                metadata.get(
                    "source_id"
                )
                or resolution.get(
                    "source_id"
                )
                or paper_dir.name
            ),

            "topic_axis": (
                metadata.get(
                    "topic_axis"
                )
                or resolution.get(
                    "topic_axis"
                )
                or paper_dir.parent.name
            ),

            "selected_format": (
                resolution.get(
                    "selected_format"
                )
            ),

            "source_url": (
                resolution.get(
                    "source_url"
                )
            ),
        },
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

    raw_path.write_text(
        parsed[
            "raw_content"
        ],
        encoding="utf-8",
    )

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

        "source_info": (
            parsed[
                "source_info"
            ]
        ),
    }

    _save_json(
        parsed_path,
        payload,
    )


# ============================================================
# Find Files
# ============================================================

def _find_txt_files() -> list[
    Path
]:
    """
    NTRS resolver v2 결과:

    data/tmp/ntrs_resolved/
        rover_autonomy/
            ID/
                paper.txt
                metadata.json
                resolution.json
    """

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            "NTRS resolved root "
            f"not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/paper.txt"
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
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    failed_count = (
        len(
            results
        )
        - success_count
    )

    total_words = sum(
        item.get(
            "word_count",
            0,
        )
        for item in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    total_chars = sum(
        item.get(
            "char_count",
            0,
        )
        for item in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    payload = {
        "total": len(
            results
        ),

        "success": (
            success_count
        ),

        "failed": (
            failed_count
        ),

        "total_words": (
            total_words
        ),

        "total_chars": (
            total_chars
        ),

        "documents": (
            results
        ),
    }

    _save_json(
        REPORT_FILE,
        payload,
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - NASA NTRS TXT Parser"
    )

    print("=" * 70)

    print(
        f"Input root : "
        f"{RESOLVED_ROOT}"
    )

    txt_files = (
        _find_txt_files()
    )

    print(
        f"TXT files  : "
        f"{len(txt_files)}"
    )

    if not txt_files:

        print()
        print(
            "[STOP] No paper.txt files found."
        )

        return

    print()

    results: list[
        dict
    ] = []

    # ========================================================
    # Parse
    # ========================================================

    for index, txt_path in enumerate(
        txt_files,
        start=1,
    ):

        paper_dir = (
            txt_path.parent
        )

        source_id = (
            paper_dir.name
        )

        topic_axis = (
            paper_dir.parent.name
        )

        print(
            "-" * 70
        )

        print(
            f"[{index}/"
            f"{len(txt_files)}]"
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
            f"{txt_path}"
        )

        print(
            "-" * 70
        )

        try:

            parsed = (
                parse_txt_file(
                    txt_path
                )
            )

            stats = (
                parsed[
                    "stats"
                ]
            )

            _save_parsed_document(
                paper_dir,
                parsed,
            )

            print(
                f"[OK] Title      : "
                f"{parsed['title']}"
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
                f"[OK] Words      : "
                f"{stats['word_count']}"
            )

            print(
                f"[OK] Chars      : "
                f"{stats['char_count']}"
            )

            print(
                f"[OK] Alpha      : "
                f"{stats['alpha_ratio']}"
            )

            print(
                "[SAVE] "
                f"{paper_dir / 'raw_content.txt'}"
            )

            results.append(
                {
                    "status": (
                        "success"
                    ),

                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "title": (
                        parsed[
                            "title"
                        ]
                    ),

                    "section_count": (
                        stats[
                            "section_count"
                        ]
                    ),

                    "paragraph_count": (
                        stats[
                            "paragraph_count"
                        ]
                    ),

                    "word_count": (
                        stats[
                            "word_count"
                        ]
                    ),

                    "char_count": (
                        stats[
                            "char_count"
                        ]
                    ),

                    "alpha_ratio": (
                        stats[
                            "alpha_ratio"
                        ]
                    ),
                }
            )

        except Exception as exc:

            print(
                f"[FAILED] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            results.append(
                {
                    "status": (
                        "failed"
                    ),

                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

        print()

    # ========================================================
    # Report
    # ========================================================

    _save_report(
        results
    )

    success_count = sum(
        1
        for item in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    failed_count = (
        len(
            results
        )
        - success_count
    )

    total_words = sum(
        item.get(
            "word_count",
            0,
        )
        for item in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    total_chars = sum(
        item.get(
            "char_count",
            0,
        )
        for item in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "NTRS TXT PARSING COMPLETED"
    )

    print(
        "=" * 70
    )

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

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()