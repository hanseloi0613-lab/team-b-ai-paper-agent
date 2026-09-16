import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - NASA NTRS Core-100 Batch Parser
# ============================================================
#
# Frozen Selection:
#
#   rover_autonomy       12
#   onboard_ai           12
#   satellite_autonomy   11
#   ------------------------
#   TOTAL                35
#
#
# Resolver result:
#
#   TXT                  26
#   PDF                   9
#   ------------------------
#   TOTAL                35
#
#
# 이 Parser는:
#
#   ntrs_core100_selected.json
#               ↓
#   선택된 35편 ONLY
#               ↓
#       resolution.json
#          ↙         ↘
#       TXT           PDF
#        ↓             ↓
#   TXT parser     PDF parser
#          ↘         ↙
#       raw_content.txt
#       parsed_document.json
#               ↓
#           35 / 35 QA
#
#
# 중요:
#
# - 기존 Pilot 15편은 건드리지 않는다.
# - ntrs_resolved 전체를 glob하지 않는다.
# - frozen selection manifest만 canonical source로 사용한다.
# - PDF는 pypdf를 사용한다.
# - OCR은 사용하지 않는다.
# - cleaning은 여기서 하지 않는다.
# - References 제거는 Cleaner에서 한다.
# ============================================================


PARSER_VERSION = "core100_v1"


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
    / "ntrs_core100_parsing_report.json"
)


# ============================================================
# Parser Minimum Gate
# ============================================================
#
# 이것은 Cleaner quality threshold가 아니다.
#
# 여기서는:
# "parser가 실질적인 text를 만들어냈는가?"
# 만 확인한다.
#
# 최종 품질 기준은 Cleaner에서 별도로 적용한다.
# ============================================================

MIN_PARSED_CHARS = 500

MIN_PARSED_WORDS = 80


# ============================================================
# Heading Vocabulary
# ============================================================

KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "related works",
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
    "design",
    "system design",
    "problem formulation",
    "experimental setup",
    "experiment",
    "experiments",
    "evaluation",
    "analysis",
    "results",
    "discussion",
    "results and discussion",
    "limitations",
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

        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:

        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:

        raise RuntimeError(
            f"Invalid JSON file: {path}"
        ) from exc


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
# Frozen Selection
# ============================================================

def _load_selection() -> dict[
    str,
    list[str],
]:

    payload = (
        _load_json(
            SELECTION_FILE
        )
    )

    # --------------------------------------------------------
    # Version
    # --------------------------------------------------------

    version = (
        payload.get(
            "selection_version"
        )
    )

    if (
        version is not None
        and version != PARSER_VERSION
    ):

        raise RuntimeError(
            "Selection version mismatch.\n"
            f"Expected: {PARSER_VERSION}\n"
            f"Found   : {version}"
        )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source = str(
        payload.get(
            "source",
            "",
        )
    ).lower()

    if (
        source
        and source != "ntrs"
    ):

        raise RuntimeError(
            f"Unexpected selection source: "
            f"{source}"
        )

    # --------------------------------------------------------
    # Canonical IDs
    # --------------------------------------------------------

    selected = (
        payload.get(
            "selected_documents"
        )
    )

    if not isinstance(
        selected,
        dict,
    ):

        raise RuntimeError(
            "selected_documents missing "
            "from ntrs_core100_selected.json."
        )

    result = {}

    seen_ids = set()

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        source_ids = (
            selected.get(
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

        normalized = []

        for source_id in source_ids:

            source_id = str(
                source_id
            ).strip()

            if not source_id:

                raise RuntimeError(
                    f"{topic_axis}: "
                    "empty source_id."
                )

            if source_id in seen_ids:

                raise RuntimeError(
                    "Cross-axis duplicate "
                    f"NTRS ID: {source_id}"
                )

            seen_ids.add(
                source_id
            )

            normalized.append(
                source_id
            )

        if (
            len(normalized)
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count}, "
                f"found {len(normalized)}."
            )

        result[
            topic_axis
        ] = normalized

    total = sum(
        len(source_ids)
        for source_ids
        in result.values()
    )

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} "
            f"selected documents, "
            f"found {total}."
        )

    return result


# ============================================================
# Basic Text Normalization
# ============================================================

def _normalize_text(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    text = (
        text
        .replace(
            "\x00",
            "",
        )
        .replace(
            "\ufeff",
            "",
        )
        .replace(
            "\xa0",
            " ",
        )
        .replace(
            "\u00ad",
            "",
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
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .replace(
            "\f",
            "\n\n",
        )
        .replace(
            "\t",
            " ",
        )
    )

    text = re.sub(
        r"[ ]{2,}",
        " ",
        text,
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    return text.strip()


def _normalize_space(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            text
            or ""
        ),
    ).strip()


def _clean_line(
    line: str,
) -> str:

    line = (
        line
        .replace(
            "\u00ad",
            "",
        )
        .replace(
            "\xa0",
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
    )

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

    text = str(
        text
        or ""
    ).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


# ============================================================
# Metadata / Resolution
# ============================================================

def _load_document_metadata(
    paper_dir: Path,
) -> tuple[
    dict,
    dict,
]:

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
        or paper_dir.name
    )

    return _normalize_space(
        str(
            title
        )
    )


def _extract_abstract(
    metadata: dict,
) -> str:

    abstract = (
        metadata.get(
            "abstract"
        )
        or ""
    )

    return _normalize_space(
        str(
            abstract
        )
    )


# ============================================================
# Noise Detection
# ============================================================

def _is_page_number(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

    if re.fullmatch(
        r"\d{1,4}",
        stripped,
    ):

        return True

    if re.fullmatch(
        r"page\s+\d+\s*(?:of\s+\d+)?",
        stripped,
        flags=re.IGNORECASE,
    ):

        return True

    return False


def _is_separator_line(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

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

    stripped = (
        line.strip()
    )

    if not stripped:

        return False

    if _is_separator_line(
        stripped
    ):

        return True

    if _is_page_number(
        stripped
    ):

        return True

    return False


# ============================================================
# Heading Detection
# ============================================================

def _strip_heading_number(
    line: str,
) -> str:

    line = str(
        line
        or ""
    ).strip()

    # 1. Introduction
    # 2.1 Navigation
    line = re.sub(
        r"^\d+(?:\.\d+)*[\.\)]?\s+",
        "",
        line,
    )

    # III. Results
    line = re.sub(
        r"^[IVXLC]+\.\s+",
        "",
        line,
    )

    # A. Architecture
    line = re.sub(
        r"^[A-Z]\.\s+",
        "",
        line,
    )

    return line.strip()


def _looks_like_sentence(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

    if not stripped:

        return False

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
        .strip(
            " :.-"
        )
    )

    return (
        normalized
        in KNOWN_HEADINGS
    )


def _is_numbered_heading(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

    if (
        len(
            stripped
        ) < 4
        or len(
            stripped
        ) > 140
    ):

        return False

    numeric_pattern = re.compile(
        r"^\d+(?:\.\d+)*"
        r"[\.\)]?"
        r"\s+"
        r".{2,120}$"
    )

    roman_pattern = re.compile(
        r"^[IVXLC]+\."
        r"\s+"
        r".{2,120}$"
    )

    letter_pattern = re.compile(
        r"^[A-Z]\."
        r"\s+"
        r".{2,120}$"
    )

    if not (
        numeric_pattern.match(
            stripped
        )
        or roman_pattern.match(
            stripped
        )
        or letter_pattern.match(
            stripped
        )
    ):

        return False

    heading_text = (
        _strip_heading_number(
            stripped
        )
    )

    if _looks_like_sentence(
        heading_text
    ):

        return False

    alpha_count = sum(
        1
        for char
        in heading_text
        if char.isalpha()
    )

    return (
        alpha_count >= 2
    )


def _is_all_caps_heading(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

    if (
        len(
            stripped
        ) < 3
        or len(
            stripped
        ) > 120
    ):

        return False

    if (
        "@" in stripped
        or "http://" in stripped.lower()
        or "https://" in stripped.lower()
    ):

        return False

    if stripped.endswith(
        (
            ".",
            ";",
            ",",
        )
    ):

        return False

    letters = [
        char
        for char
        in stripped
        if char.isalpha()
    ]

    if len(
        letters
    ) < 3:

        return False

    uppercase_ratio = (
        sum(
            1
            for char
            in letters
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

    return (
        word_count <= 14
    )


def _is_title_case_heading(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

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

    meaningful = [
        word
        for word
        in words
        if word.lower()
        not in allowed_lower
    ]

    if not meaningful:

        return False

    capitalized = sum(
        1
        for word
        in meaningful
        if (
            word[0].isupper()
            or word.isupper()
        )
    )

    ratio = (
        capitalized
        / len(
            meaningful
        )
    )

    if ratio < 0.80:

        return False

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
        "autonomy",
        "operations",
        "simulation",
        "performance",
        "algorithm",
    }

    return any(
        word.lower()
        in heading_terms
        for word
        in words
    )


def _is_heading(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

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

    stripped = (
        line.strip()
    )

    numeric_match = re.match(
        r"^(\d+(?:\.\d+)*)",
        stripped,
    )

    if numeric_match:

        number = (
            numeric_match.group(
                1
            )
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

    # Roman / A. / normal heading
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

    stripped = (
        text.strip()
    )

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
        for symbol
        in (
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
        for line
        in lines
        if line.strip()
    )

    # naviga- tion -> navigation
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
# Text -> Blocks
# ============================================================

def _extract_blocks(
    text: str,
    title: str,
) -> list[dict]:

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

        # blank line은 paragraph boundary
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
        # Blank
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
        # Figure / Table
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
        # List
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
        # Equation
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
        .strip(
            " :.-"
        )
    )

    return (
        normalized
        == "abstract"
    )


def _blocks_to_sections(
    blocks: list[dict],
) -> list[dict]:

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
                "heading": (
                    heading
                ),
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

        useful_blocks = [
            block
            for block
            in blocks
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
    *,
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:

    output: list[str] = []

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Abstract
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sections
    # --------------------------------------------------------

    for section in sections:

        heading = str(
            section.get(
                "heading",
                "Document Body",
            )
        ).strip()

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
        for char
        in text
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
    *,
    page_count: int | None = None,
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

    stats = {
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

    if page_count is not None:

        stats[
            "page_count"
        ] = page_count

    return stats


# ============================================================
# Common Parse Builder
# ============================================================

def _build_parsed_result(
    *,
    paper_dir: Path,
    source_text: str,
    parser_mode: str,
    page_count: int | None = None,
) -> dict:

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
            page_count=(
                page_count
            ),
        )
    )

    source_id = (
        metadata.get(
            "source_id"
        )
        or resolution.get(
            "source_id"
        )
        or paper_dir.name
    )

    topic_axis = (
        metadata.get(
            "topic_axis"
        )
        or resolution.get(
            "topic_axis"
        )
        or paper_dir.parent.name
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
            "source": (
                "ntrs"
            ),
            "source_id": (
                source_id
            ),
            "topic_axis": (
                topic_axis
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
            "parser_mode": (
                parser_mode
            ),
            "parser_version": (
                PARSER_VERSION
            ),
            "repair": (
                resolution.get(
                    "repair"
                )
            ),
        },
    }


# ============================================================
# TXT Parser
# ============================================================

def parse_txt_file(
    txt_path: Path,
) -> dict:

    source_text = (
        txt_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    return (
        _build_parsed_result(
            paper_dir=(
                txt_path.parent
            ),
            source_text=(
                source_text
            ),
            parser_mode=(
                "ntrs_txt"
            ),
        )
    )


# ============================================================
# PDF Extraction
# ============================================================

def _extract_pdf_pages(
    pdf_path: Path,
) -> list[str]:

    reader = (
        PdfReader(
            str(
                pdf_path
            )
        )
    )

    pages = []

    for page in reader.pages:

        try:

            text = (
                page.extract_text()
                or ""
            )

        except Exception:

            text = ""

        # ----------------------------------------------------
        # line-end hyphenation
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

        text = (
            text
            .replace(
                "\u00ad",
                "",
            )
            .replace(
                "\ufb01",
                "fi",
            )
            .replace(
                "\ufb02",
                "fl",
            )
        )

        pages.append(
            text
        )

    return pages


# ============================================================
# Repeated PDF Headers / Footers
# ============================================================

def _detect_repeated_pdf_lines(
    pages: list[str],
) -> set[str]:

    counter = Counter()

    for page_text in pages:

        unique_page_lines = set()

        for raw_line in (
            page_text.splitlines()
        ):

            line = (
                _clean_line(
                    raw_line
                )
            )

            if not line:

                continue

            # 긴 본문은 반복돼도 header/footer로 제거하지 않는다.
            if len(
                line
            ) > 100:

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

    threshold = max(
        3,
        len(
            pages
        ) // 2,
    )

    return {
        line
        for (
            line,
            count,
        )
        in counter.items()
        if count >= threshold
    }


def _pdf_pages_to_text(
    pages: list[str],
) -> str:

    repeated_lines = (
        _detect_repeated_pdf_lines(
            pages
        )
    )

    document_lines = []

    for page_text in pages:

        page_lines = []

        for raw_line in (
            page_text.splitlines()
        ):

            line = (
                _clean_line(
                    raw_line
                )
            )

            if not line:

                # 원래 PDF에 blank가 있으면
                # paragraph boundary 유지
                if (
                    page_lines
                    and page_lines[
                        -1
                    ] != ""
                ):

                    page_lines.append(
                        ""
                    )

                continue

            if (
                line
                in repeated_lines
            ):

                continue

            if _is_page_number(
                line
            ):

                continue

            page_lines.append(
                line
            )

        document_lines.extend(
            page_lines
        )

        # page boundary
        document_lines.extend(
            [
                "",
                "",
            ]
        )

    return "\n".join(
        document_lines
    ).strip()


# ============================================================
# PDF Parser
# ============================================================

def parse_pdf_file(
    pdf_path: Path,
) -> dict:

    pages = (
        _extract_pdf_pages(
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
        for page
        in pages
    )

    if extracted_char_count < 500:

        raise RuntimeError(
            "PDF text extraction produced "
            "too little text."
        )

    source_text = (
        _pdf_pages_to_text(
            pages
        )
    )

    return (
        _build_parsed_result(
            paper_dir=(
                pdf_path.parent
            ),
            source_text=(
                source_text
            ),
            parser_mode=(
                "ntrs_pdf_pypdf"
            ),
            page_count=(
                len(
                    pages
                )
            ),
        )
    )


# ============================================================
# Parsed Result Validation
# ============================================================

def _validate_parsed(
    parsed: dict,
) -> None:

    raw_content = str(
        parsed.get(
            "raw_content",
            "",
        )
    ).strip()

    stats = (
        parsed.get(
            "stats",
            {}
        )
    )

    if not raw_content:

        raise RuntimeError(
            "Parsed raw_content is empty."
        )

    char_count = int(
        stats.get(
            "char_count",
            0,
        )
        or 0
    )

    word_count = int(
        stats.get(
            "word_count",
            0,
        )
        or 0
    )

    section_count = int(
        stats.get(
            "section_count",
            0,
        )
        or 0
    )

    if (
        char_count
        < MIN_PARSED_CHARS
    ):

        raise RuntimeError(
            "Parsed content too short: "
            f"{char_count} chars."
        )

    if (
        word_count
        < MIN_PARSED_WORDS
    ):

        raise RuntimeError(
            "Parsed content too short: "
            f"{word_count} words."
        )

    if section_count < 1:

        raise RuntimeError(
            "No parsed sections."
        )


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


def _remove_stale_outputs(
    paper_dir: Path,
) -> None:

    for filename in (
        "raw_content.txt",
        "parsed_document.json",
    ):

        path = (
            paper_dir
            / filename
        )

        if path.exists():

            try:

                path.unlink()

            except OSError:

                pass


# ============================================================
# Resolve Canonical Source File
# ============================================================

def _resolve_input_file(
    *,
    topic_axis: str,
    source_id: str,
) -> tuple[
    str,
    Path,
    dict,
]:

    paper_dir = (
        RESOLVED_ROOT
        / topic_axis
        / source_id
    )

    if not paper_dir.exists():

        raise FileNotFoundError(
            "Resolved paper directory "
            f"not found: {paper_dir}"
        )

    resolution_path = (
        paper_dir
        / "resolution.json"
    )

    resolution = (
        _load_json(
            resolution_path
        )
    )

    if not resolution.get(
        "success",
        False,
    ):

        raise RuntimeError(
            f"{source_id}: "
            "resolution.success is not True."
        )

    resolution_source_id = str(
        resolution.get(
            "source_id",
            "",
        )
    ).strip()

    if (
        resolution_source_id
        and resolution_source_id
        != source_id
    ):

        raise RuntimeError(
            f"{source_id}: "
            "resolution source_id mismatch: "
            f"{resolution_source_id}"
        )

    resolution_axis = str(
        resolution.get(
            "topic_axis",
            "",
        )
    ).strip()

    if (
        resolution_axis
        and resolution_axis
        != topic_axis
    ):

        raise RuntimeError(
            f"{source_id}: "
            "resolution topic_axis mismatch: "
            f"{resolution_axis}"
        )

    selected_format = str(
        resolution.get(
            "selected_format",
            "",
        )
    ).lower().strip()

    if selected_format not in {
        "txt",
        "original_text",
        "pdf",
    }:

        raise RuntimeError(
            f"{source_id}: "
            "unsupported selected_format: "
            f"{selected_format}"
        )

    # --------------------------------------------------------
    # Resolver canonical local_file
    # --------------------------------------------------------

    local_file_value = (
        resolution.get(
            "local_file"
        )
    )

    local_file = None

    if local_file_value:

        local_file = Path(
            str(
                local_file_value
            )
        )

        if not local_file.is_absolute():

            local_file = (
                paper_dir
                / local_file
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if (
        local_file is None
        or not local_file.exists()
    ):

        if selected_format in {
            "txt",
            "original_text",
        }:

            fallback = (
                paper_dir
                / "paper.txt"
            )

        else:

            fallback = (
                paper_dir
                / "paper.pdf"
            )

        if fallback.exists():

            local_file = (
                fallback
            )

    if (
        local_file is None
        or not local_file.exists()
    ):

        raise FileNotFoundError(
            f"{source_id}: "
            "resolved local file missing."
        )

    return (
        selected_format,
        local_file,
        resolution,
    )


# ============================================================
# Parse One Frozen Document
# ============================================================

def _parse_selected_document(
    *,
    topic_axis: str,
    source_id: str,
) -> dict:

    (
        selected_format,
        input_path,
        resolution,
    ) = (
        _resolve_input_file(
            topic_axis=(
                topic_axis
            ),
            source_id=(
                source_id
            ),
        )
    )

    paper_dir = (
        RESOLVED_ROOT
        / topic_axis
        / source_id
    )

    if selected_format in {
        "txt",
        "original_text",
    }:

        parsed = (
            parse_txt_file(
                input_path
            )
        )

        parser_type = (
            "TXT"
        )

    elif (
        selected_format
        == "pdf"
    ):

        parsed = (
            parse_pdf_file(
                input_path
            )
        )

        parser_type = (
            "PDF"
        )

    else:

        raise RuntimeError(
            f"Unsupported format: "
            f"{selected_format}"
        )

    _validate_parsed(
        parsed
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

    return {
        "status": (
            "success"
        ),
        "topic_axis": (
            topic_axis
        ),
        "source_id": (
            source_id
        ),
        "selected_format": (
            selected_format
        ),
        "parser_type": (
            parser_type
        ),
        "input_file": (
            str(
                input_path
            )
        ),
        "title": (
            parsed[
                "title"
            ]
        ),
        "section_count": (
            stats.get(
                "section_count",
                0,
            )
        ),
        "paragraph_count": (
            stats.get(
                "paragraph_count",
                0,
            )
        ),
        "word_count": (
            stats.get(
                "word_count",
                0,
            )
        ),
        "char_count": (
            stats.get(
                "char_count",
                0,
            )
        ),
        "alpha_ratio": (
            stats.get(
                "alpha_ratio",
                0.0,
            )
        ),
        "page_count": (
            stats.get(
                "page_count"
            )
        ),
        "repair": (
            resolution.get(
                "repair"
            )
        ),
    }


# ============================================================
# Final Output Verification
# ============================================================

def _verify_output_files(
    *,
    topic_axis: str,
    source_id: str,
) -> None:

    paper_dir = (
        RESOLVED_ROOT
        / topic_axis
        / source_id
    )

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    parsed_path = (
        paper_dir
        / "parsed_document.json"
    )

    if not raw_path.exists():

        raise RuntimeError(
            f"{source_id}: "
            "raw_content.txt missing "
            "after parser."
        )

    if not parsed_path.exists():

        raise RuntimeError(
            f"{source_id}: "
            "parsed_document.json missing "
            "after parser."
        )

    if not raw_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip():

        raise RuntimeError(
            f"{source_id}: "
            "raw_content.txt is empty."
        )

    parsed = (
        _load_json(
            parsed_path
        )
    )

    if not parsed.get(
        "sections"
    ):

        raise RuntimeError(
            f"{source_id}: "
            "parsed_document.json has "
            "no sections."
        )


# ============================================================
# Report
# ============================================================

def _save_report(
    *,
    results: list[dict],
    format_counts: dict[str, int],
    parser_counts: dict[str, int],
    axis_success: dict[str, int],
) -> None:

    success_count = sum(
        1
        for result
        in results
        if (
            result.get(
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

    successful = [
        result
        for result
        in results
        if (
            result.get(
                "status"
            )
            == "success"
        )
    ]

    total_words = sum(
        int(
            result.get(
                "word_count",
                0,
            )
            or 0
        )
        for result
        in successful
    )

    total_chars = sum(
        int(
            result.get(
                "char_count",
                0,
            )
            or 0
        )
        for result
        in successful
    )

    word_counts = [
        int(
            result.get(
                "word_count",
                0,
            )
            or 0
        )
        for result
        in successful
    ]

    char_counts = [
        int(
            result.get(
                "char_count",
                0,
            )
            or 0
        )
        for result
        in successful
    ]

    payload = {
        "parser_version": (
            PARSER_VERSION
        ),
        "selection_file": (
            str(
                SELECTION_FILE
            )
        ),
        "expected_total": (
            EXPECTED_TOTAL
        ),
        "selected": (
            len(
                results
            )
        ),
        "success": (
            success_count
        ),
        "failed": (
            failed_count
        ),
        "resolved_formats": (
            format_counts
        ),
        "parser_success": (
            parser_counts
        ),
        "axis_success": (
            axis_success
        ),
        "total_words": (
            total_words
        ),
        "total_chars": (
            total_chars
        ),
        "word_range": (
            {
                "min": (
                    min(
                        word_counts
                    )
                    if word_counts
                    else 0
                ),
                "max": (
                    max(
                        word_counts
                    )
                    if word_counts
                    else 0
                ),
            }
        ),
        "char_range": (
            {
                "min": (
                    min(
                        char_counts
                    )
                    if char_counts
                    else 0
                ),
                "max": (
                    max(
                        char_counts
                    )
                    if char_counts
                    else 0
                ),
            }
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
    print("=" * 78)

    print(
        "TEAM B - NASA NTRS "
        "Core-100 Batch Parser"
    )

    print("=" * 78)

    print(
        f"Parser version : "
        f"{PARSER_VERSION}"
    )

    print(
        f"Selection file : "
        f"{SELECTION_FILE}"
    )

    print(
        f"Input root     : "
        f"{RESOLVED_ROOT}"
    )

    print(
        f"Report         : "
        f"{REPORT_FILE}"
    )

    # ========================================================
    # Frozen Selection
    # ========================================================

    selection = (
        _load_selection()
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
            selection[
                topic_axis
            ]
        )

        print(
            f"  "
            f"{topic_axis:22} : "
            f"{actual} / "
            f"{expected_count}"
        )

    print(
        f"  "
        f"{'TOTAL':22} : "
        f"{sum(len(x) for x in selection.values())}"
    )

    # ========================================================
    # Counters
    # ========================================================

    results = []

    format_counts = {
        "txt": 0,
        "original_text": 0,
        "pdf": 0,
    }

    parser_counts = {
        "TXT": 0,
        "PDF": 0,
    }

    axis_success = {
        topic_axis: 0
        for topic_axis
        in EXPECTED_COUNTS
    }

    # ========================================================
    # Parse exactly selected 35
    # ========================================================

    global_index = 0

    for topic_axis in EXPECTED_COUNTS:

        for source_id in (
            selection[
                topic_axis
            ]
        ):

            global_index += 1

            print()
            print("-" * 78)

            print(
                f"[{global_index}/"
                f"{EXPECTED_TOTAL}]"
            )

            print(
                f"[AXIS] "
                f"{topic_axis}"
            )

            print(
                f"[ID]   "
                f"{source_id}"
            )

            try:

                (
                    selected_format,
                    input_path,
                    _resolution,
                ) = (
                    _resolve_input_file(
                        topic_axis=(
                            topic_axis
                        ),
                        source_id=(
                            source_id
                        ),
                    )
                )

                print(
                    f"[FMT]  "
                    f"{selected_format}"
                )

                print(
                    f"[FILE] "
                    f"{input_path}"
                )

                result = (
                    _parse_selected_document(
                        topic_axis=(
                            topic_axis
                        ),
                        source_id=(
                            source_id
                        ),
                    )
                )

                _verify_output_files(
                    topic_axis=(
                        topic_axis
                    ),
                    source_id=(
                        source_id
                    ),
                )

                format_counts[
                    selected_format
                ] += 1

                parser_counts[
                    result[
                        "parser_type"
                    ]
                ] += 1

                axis_success[
                    topic_axis
                ] += 1

                print(
                    f"[OK] Title      : "
                    f"{result['title']}"
                )

                if (
                    result.get(
                        "page_count"
                    )
                    is not None
                ):

                    print(
                        f"[OK] Pages      : "
                        f"{result['page_count']}"
                    )

                print(
                    f"[OK] Sections   : "
                    f"{result['section_count']}"
                )

                print(
                    f"[OK] Paragraphs : "
                    f"{result['paragraph_count']}"
                )

                print(
                    f"[OK] Words      : "
                    f"{result['word_count']}"
                )

                print(
                    f"[OK] Chars      : "
                    f"{result['char_count']}"
                )

                print(
                    f"[OK] Alpha      : "
                    f"{result['alpha_ratio']}"
                )

                print(
                    "[SAVE] "
                    f"{RESOLVED_ROOT / topic_axis / source_id / 'raw_content.txt'}"
                )

                print(
                    "[SAVE] "
                    f"{RESOLVED_ROOT / topic_axis / source_id / 'parsed_document.json'}"
                )

                results.append(
                    result
                )

            except Exception as exc:

                # stale parser output가 남아
                # 다음 Cleaner가 잘못 통과하는 것을 방지
                _remove_stale_outputs(
                    RESOLVED_ROOT
                    / topic_axis
                    / source_id
                )

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

    # ========================================================
    # Result Stats
    # ========================================================

    success_count = sum(
        1
        for result
        in results
        if (
            result.get(
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

    successful = [
        result
        for result
        in results
        if (
            result.get(
                "status"
            )
            == "success"
        )
    ]

    total_words = sum(
        int(
            result.get(
                "word_count",
                0,
            )
            or 0
        )
        for result
        in successful
    )

    total_chars = sum(
        int(
            result.get(
                "char_count",
                0,
            )
            or 0
        )
        for result
        in successful
    )

    word_counts = [
        int(
            result.get(
                "word_count",
                0,
            )
            or 0
        )
        for result
        in successful
    ]

    char_counts = [
        int(
            result.get(
                "char_count",
                0,
            )
            or 0
        )
        for result
        in successful
    ]

    # ========================================================
    # Report
    # ========================================================

    _save_report(
        results=(
            results
        ),
        format_counts=(
            format_counts
        ),
        parser_counts=(
            parser_counts
        ),
        axis_success=(
            axis_success
        ),
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 78)

    print(
        "NTRS CORE-100 BATCH "
        "PARSING COMPLETED"
    )

    print("=" * 78)

    print(
        f"Selected : "
        f"{len(results)}"
    )

    print(
        f"Success  : "
        f"{success_count}"
    )

    print(
        f"Failed   : "
        f"{failed_count}"
    )

    print()

    print(
        "Resolved formats:"
    )

    print(
        f"  TXT           : "
        f"{format_counts['txt']}"
    )

    print(
        f"  Original text : "
        f"{format_counts['original_text']}"
    )

    print(
        f"  PDF           : "
        f"{format_counts['pdf']}"
    )

    print()

    print(
        "Parser success:"
    )

    print(
        f"  TXT : "
        f"{parser_counts['TXT']}"
    )

    print(
        f"  PDF : "
        f"{parser_counts['PDF']}"
    )

    print()

    print(
        "Axis QA:"
    )

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        print(
            f"  "
            f"{topic_axis:22} : "
            f"{axis_success[topic_axis]} / "
            f"{expected_count}"
        )

    print()

    print(
        f"Total words : "
        f"{total_words}"
    )

    print(
        f"Total chars : "
        f"{total_chars}"
    )

    if word_counts:

        print(
            f"Word range  : "
            f"{min(word_counts)} "
            f"~ "
            f"{max(word_counts)}"
        )

    if char_counts:

        print(
            f"Char range  : "
            f"{min(char_counts)} "
            f"~ "
            f"{max(char_counts)}"
        )

    print()

    print(
        f"Report      : "
        f"{REPORT_FILE}"
    )

    print("=" * 78)

    # ========================================================
    # Hard Gate
    # ========================================================

    all_axes_pass = all(
        axis_success[
            topic_axis
        ]
        == expected_count
        for (
            topic_axis,
            expected_count,
        )
        in EXPECTED_COUNTS.items()
    )

    all_formats_accounted = (
        sum(
            format_counts.values()
        )
        == EXPECTED_TOTAL
    )

    parser_count_ok = (
        sum(
            parser_counts.values()
        )
        == EXPECTED_TOTAL
    )

    if (
        len(
            results
        )
        == EXPECTED_TOTAL
        and success_count
        == EXPECTED_TOTAL
        and failed_count
        == 0
        and all_axes_pass
        and all_formats_accounted
        and parser_count_ok
    ):

        print(
            "[PASS] All 35 selected "
            "NTRS papers parsed."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Run NTRS Cleaner for "
            "these 35 documents only."
        )

        print("=" * 78)

        return

    print(
        "[CHECK] Parsing Gate failed."
    )

    print()

    print(
        "Do NOT run Cleaner yet."
    )

    print(
        "Inspect only the failed "
        "NTRS documents above."
    )

    print("=" * 78)

    raise RuntimeError(
        "NTRS Core-100 Parser "
        f"Gate failed: "
        f"{success_count}/"
        f"{EXPECTED_TOTAL} parsed."
    )


if __name__ == "__main__":
    main()