import hashlib
import json
import re
import unicodedata
from pathlib import Path

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - NASA NTRS Core-100 Cleaner
# ============================================================
#
# Pipeline:
#
# ntrs_core100_selected.json
#          ↓
# frozen new 35 ONLY
#          ↓
# parsed_document.json
# raw_content.txt
#          ↓
# normalize
# remove obvious residue
# remove references/back-matter
# preserve academic sentences
#          ↓
# clean_content.txt
# cleaned_document.json
#          ↓
# quality gate
#          ↓
# SHA-256 duplicate QA
#          ↓
# 35 / 35 PASS
#
#
# IMPORTANT
#
# - Pilot 15 documents are NOT processed again.
# - Do NOT glob every parsed_document.json.
# - Frozen manifest is the canonical source.
# - Thresholds are NOT lowered for short documents.
# - Cleaning is conservative.
# ============================================================


CLEANER_VERSION = "core100_v1"

NORMALIZATION_VERSION = "v1"


# ============================================================
# Expected Frozen Selection
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
    / "ntrs_core100_cleaning_report.json"
)


# ============================================================
# Quality Thresholds
# ============================================================
#
# arXiv Core와 동일한 baseline.
#
# 낮추지 않는다.
# ============================================================

MIN_CHAR_COUNT = 5_000

MIN_WORD_COUNT = 800

MIN_ALPHA_RATIO = 0.45

MIN_SECTION_COUNT = 2


# ============================================================
# Sections Removed From Clean Corpus
# ============================================================

REFERENCE_HEADINGS = {
    "reference",
    "references",
    "bibliography",
    "bibliographies",
    "literature cited",
    "works cited",
}


REMOVE_SECTION_HEADINGS = {
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",

    "funding",
    "funding information",
    "funding statement",

    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "declaration of competing interest",

    "author contribution",
    "author contributions",
    "authors contributions",

    "data availability",
    "data availability statement",

    "ethics statement",
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

    # ========================================================
    # Version
    # ========================================================

    selection_version = (
        payload.get(
            "selection_version"
        )
    )

    if (
        selection_version is not None
        and selection_version
        != CLEANER_VERSION
    ):

        raise RuntimeError(
            "Selection version mismatch.\n"
            f"Expected: {CLEANER_VERSION}\n"
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
    ).strip().lower()

    if (
        source
        and source != "ntrs"
    ):

        raise RuntimeError(
            "Unexpected selection source: "
            f"{source}"
        )

    # ========================================================
    # Frozen IDs
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
            "from NTRS selection manifest."
        )

    result: dict[
        str,
        list[str],
    ] = {}

    global_seen = set()

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
                f"{topic_axis}: "
                "selection is not a list."
            )

        normalized_ids = []

        for raw_source_id in source_ids:

            source_id = str(
                raw_source_id
            ).strip()

            if not source_id:

                raise RuntimeError(
                    f"{topic_axis}: "
                    "empty source_id."
                )

            if source_id in global_seen:

                raise RuntimeError(
                    "Cross-axis duplicate "
                    f"source_id: {source_id}"
                )

            global_seen.add(
                source_id
            )

            normalized_ids.append(
                source_id
            )

        if (
            len(
                normalized_ids
            )
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count}, "
                f"found "
                f"{len(normalized_ids)}."
            )

        result[
            topic_axis
        ] = normalized_ids

    total = sum(
        len(
            source_ids
        )
        for source_ids
        in result.values()
    )

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} "
            f"selected NTRS documents, "
            f"found {total}."
        )

    return result


# ============================================================
# Preflight Frozen Parsed Documents
# ============================================================

def _validate_parsed_identity(
    *,
    parsed_path: Path,
    topic_axis: str,
    source_id: str,
) -> None:

    parsed = (
        _load_json(
            parsed_path
        )
    )

    source_info = (
        parsed.get(
            "source_info",
            {},
        )
    )

    if not isinstance(
        source_info,
        dict,
    ):

        source_info = {}

    parsed_source = str(
        source_info.get(
            "source",
            "",
        )
    ).strip().lower()

    if (
        parsed_source
        and parsed_source != "ntrs"
    ):

        raise RuntimeError(
            f"{source_id}: "
            "parsed source mismatch: "
            f"{parsed_source}"
        )

    parsed_source_id = str(
        source_info.get(
            "source_id",
            "",
        )
    ).strip()

    if (
        parsed_source_id
        and parsed_source_id
        != source_id
    ):

        raise RuntimeError(
            f"{source_id}: "
            "parsed source_id mismatch: "
            f"{parsed_source_id}"
        )

    parsed_axis = str(
        source_info.get(
            "topic_axis",
            "",
        )
    ).strip()

    if (
        parsed_axis
        and parsed_axis
        != topic_axis
    ):

        raise RuntimeError(
            f"{source_id}: "
            "parsed topic_axis mismatch: "
            f"{parsed_axis}"
        )

    sections = (
        parsed.get(
            "sections"
        )
    )

    if not isinstance(
        sections,
        list,
    ):

        raise RuntimeError(
            f"{source_id}: "
            "parsed sections is not a list."
        )

    if not sections:

        raise RuntimeError(
            f"{source_id}: "
            "parsed sections is empty."
        )

    raw_path = (
        parsed_path.parent
        / "raw_content.txt"
    )

    if not raw_path.exists():

        raise FileNotFoundError(
            f"{source_id}: "
            f"raw_content.txt missing: "
            f"{raw_path}"
        )

    raw_content = (
        raw_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    if not raw_content.strip():

        raise RuntimeError(
            f"{source_id}: "
            "raw_content.txt is empty."
        )


def _build_selected_documents(
    selection: dict[
        str,
        list[str],
    ],
) -> list[
    tuple[
        str,
        str,
        Path,
    ]
]:

    selected = []

    for topic_axis in EXPECTED_COUNTS:

        for source_id in (
            selection[
                topic_axis
            ]
        ):

            paper_dir = (
                RESOLVED_ROOT
                / topic_axis
                / source_id
            )

            parsed_path = (
                paper_dir
                / "parsed_document.json"
            )

            if not paper_dir.exists():

                raise FileNotFoundError(
                    "Resolved document directory "
                    f"missing: {paper_dir}"
                )

            if not parsed_path.exists():

                raise FileNotFoundError(
                    f"{source_id}: "
                    "parsed_document.json missing."
                )

            _validate_parsed_identity(
                parsed_path=(
                    parsed_path
                ),
                topic_axis=(
                    topic_axis
                ),
                source_id=(
                    source_id
                ),
            )

            selected.append(
                (
                    topic_axis,
                    source_id,
                    parsed_path,
                )
            )

    if (
        len(
            selected
        )
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            "Parsed document preflight "
            f"expected {EXPECTED_TOTAL}, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# Text Normalization
# ============================================================

def _normalize_text(
    text: str,
) -> str:
    """
    의미를 변경하지 않는 범위에서만 normalize.

    하지 않는 것:
    - lowercase
    - stemming
    - stopword 제거
    - 숫자 제거
    - 문장 재작성

    Transformer/RAG에 원래 학술 문장을 보존한다.
    """

    if not text:

        return ""

    text = unicodedata.normalize(
        "NFC",
        str(
            text
        ),
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
        "\u2009",
        " ",
    )

    text = text.replace(
        "\u202f",
        " ",
    )

    text = text.replace(
        "\u00ad",
        "",
    )

    text = text.replace(
        "\ufb01",
        "fi",
    )

    text = text.replace(
        "\ufb02",
        "fl",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = text.replace(
        "\t",
        " ",
    )

    # --------------------------------------------------------
    # line 내부 여러 space
    # --------------------------------------------------------

    text = re.sub(
        r"[ ]{2,}",
        " ",
        text,
    )

    # --------------------------------------------------------
    # punctuation 앞 이상한 whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    # --------------------------------------------------------
    # 괄호 내부 이상한 whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\(\s+",
        "(",
        text,
    )

    text = re.sub(
        r"\s+\)",
        ")",
        text,
    )

    # --------------------------------------------------------
    # 과도한 blank line
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# Heading Normalization
# ============================================================

def _normalize_heading(
    heading: str,
) -> str:

    heading = (
        _normalize_text(
            heading
        )
        .strip()
    )

    # ========================================================
    # Leading numbering
    #
    # 1. Introduction
    # 2.3 Results
    # VII. REFERENCES
    # A. Method
    # ========================================================

    heading = re.sub(
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
        heading,
        flags=re.IGNORECASE,
    )

    heading = (
        heading
        .lower()
        .strip()
        .strip(
            " .:-"
        )
    )

    heading = re.sub(
        r"\s+",
        " ",
        heading,
    )

    return heading


# ============================================================
# Section Classification
# ============================================================

def _is_reference_heading(
    heading: str,
) -> bool:

    normalized = (
        _normalize_heading(
            heading
        )
    )

    if (
        normalized
        in REFERENCE_HEADINGS
    ):

        return True

    if normalized.startswith(
        "references "
    ):

        return True

    if normalized.startswith(
        "bibliography "
    ):

        return True

    return False


def _should_remove_section(
    heading: str,
) -> bool:

    normalized = (
        _normalize_heading(
            heading
        )
    )

    return (
        normalized
        in REMOVE_SECTION_HEADINGS
    )


# ============================================================
# Noise Detection
# ============================================================

def _is_numeric_noise(
    text: str,
) -> bool:

    stripped = (
        text.strip()
    )

    if not stripped:

        return True

    # 순수 page number
    if re.fullmatch(
        r"\d{1,5}",
        stripped,
    ):

        return True

    # Page 1 / Page 1 of 10
    if re.fullmatch(
        r"page\s+\d+(?:\s+of\s+\d+)?",
        stripped,
        flags=re.IGNORECASE,
    ):

        return True

    return False


def _is_separator_noise(
    text: str,
) -> bool:

    stripped = (
        text.strip()
    )

    return bool(
        re.fullmatch(
            r"[-_=*•·.]{3,}",
            stripped,
        )
    )


def _is_probable_junk(
    text: str,
    block_type: str,
) -> bool:
    """
    확실한 residue만 제거한다.
    애매한 문장은 보존한다.
    """

    stripped = (
        _normalize_text(
            text
        )
    )

    if not stripped:

        return True

    if _is_numeric_noise(
        stripped
    ):

        return True

    if _is_separator_noise(
        stripped
    ):

        return True

    # --------------------------------------------------------
    # 너무 짧은 figure caption
    # --------------------------------------------------------

    if (
        block_type
        == "figure_caption"
        and len(
            stripped
        )
        < 20
    ):

        return True

    # --------------------------------------------------------
    # 매우 짧고 영문 정보량도 거의 없는 paragraph
    # --------------------------------------------------------

    if (
        block_type
        == "paragraph"
        and len(
            stripped
        )
        < 8
    ):

        alpha_count = sum(
            1
            for char
            in stripped
            if char.isalpha()
        )

        if alpha_count < 3:

            return True

    return False


# ============================================================
# Figure / Table Caption
# ============================================================

def _clean_caption(
    text: str,
) -> str:
    """
    Caption 내용은 유지하고
    Fig. 1 / Figure 2 / Table IV 같은 label만 제거한다.
    """

    text = (
        _normalize_text(
            text
        )
    )

    text = re.sub(
        (
            r"^(?:"
            r"fig(?:ure)?\.?"
            r"|table"
            r")"
            r"\s*"
            r"[A-Za-z0-9IVXLC\-]+"
            r"\s*"
            r"[\.:]?"
            r"\s*"
        ),
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ============================================================
# Clean One Block
# ============================================================

def _clean_block(
    block: dict,
) -> dict | None:

    block_type = str(
        block.get(
            "type",
            "paragraph",
        )
    )

    text = (
        _normalize_text(
            block.get(
                "text",
                "",
            )
        )
    )

    # parser에서 bibliography marker가 있는 경우
    if block.get(
        "in_bibliography",
        False,
    ):

        return None

    if _is_probable_junk(
        text,
        block_type,
    ):

        return None

    # --------------------------------------------------------
    # Figure/Table caption
    # --------------------------------------------------------

    if (
        block_type
        == "figure_caption"
    ):

        text = (
            _clean_caption(
                text
            )
        )

        if (
            len(
                text
            )
            < 20
        ):

            return None

    # --------------------------------------------------------
    # Equation
    #
    # 공격적인 수식 정제는 하지 않는다.
    # --------------------------------------------------------

    elif (
        block_type
        == "equation"
    ):

        text = (
            _normalize_text(
                text
            )
        )

    else:

        text = (
            _normalize_text(
                text
            )
        )

    if not text:

        return None

    return {
        "type": (
            block_type
        ),
        "text": (
            text
        ),
    }


# ============================================================
# Consecutive Duplicate Removal
# ============================================================

def _remove_consecutive_duplicates(
    blocks: list[dict],
) -> list[dict]:
    """
    Global dedup은 하지 않는다.

    같은 block이 바로 연속해서
    중복되는 경우만 제거한다.
    """

    result = []

    previous_key = None

    for block in blocks:

        key = (
            block.get(
                "type"
            ),
            block.get(
                "text"
            ),
        )

        if (
            key
            == previous_key
        ):

            continue

        result.append(
            block
        )

        previous_key = key

    return result


# ============================================================
# Clean Sections
# ============================================================

def _clean_sections(
    sections: list[dict],
) -> tuple[
    list[dict],
    dict,
]:

    clean_sections = []

    removed_sections = []

    references_removed = False

    removed_blocks = 0

    original_section_count = len(
        sections
    )

    for section in sections:

        if not isinstance(
            section,
            dict,
        ):

            continue

        heading = (
            _normalize_text(
                section.get(
                    "heading",
                    "Document Body",
                )
            )
        )

        # ====================================================
        # References Boundary
        # ====================================================
        #
        # 논문 후단의 bibliography가 시작되면
        # 그 뒤 전체를 clean body에서 제외.
        # ====================================================

        if (
            section.get(
                "in_bibliography",
                False,
            )
            or _is_reference_heading(
                heading
            )
        ):

            references_removed = True

            removed_sections.append(
                {
                    "heading": (
                        heading
                    ),
                    "reason": (
                        "references_boundary"
                    ),
                }
            )

            break

        # ====================================================
        # Other back-matter
        # ====================================================

        if _should_remove_section(
            heading
        ):

            removed_sections.append(
                {
                    "heading": (
                        heading
                    ),
                    "reason": (
                        "non_body_section"
                    ),
                }
            )

            continue

        blocks = (
            section.get(
                "blocks",
                [],
            )
        )

        if not isinstance(
            blocks,
            list,
        ):

            continue

        clean_blocks = []

        for block in blocks:

            if not isinstance(
                block,
                dict,
            ):

                removed_blocks += 1

                continue

            cleaned = (
                _clean_block(
                    block
                )
            )

            if cleaned is None:

                removed_blocks += 1

                continue

            clean_blocks.append(
                cleaned
            )

        before_dedup = len(
            clean_blocks
        )

        clean_blocks = (
            _remove_consecutive_duplicates(
                clean_blocks
            )
        )

        removed_blocks += (
            before_dedup
            - len(
                clean_blocks
            )
        )

        if not clean_blocks:

            continue

        clean_sections.append(
            {
                "heading": (
                    heading
                    or "Document Body"
                ),
                "level": (
                    section.get(
                        "level",
                        2,
                    )
                ),
                "blocks": (
                    clean_blocks
                ),
            }
        )

    info = {
        "original_section_count": (
            original_section_count
        ),
        "clean_section_count": (
            len(
                clean_sections
            )
        ),
        "references_removed": (
            references_removed
        ),
        "removed_blocks": (
            removed_blocks
        ),
        "removed_sections": (
            removed_sections
        ),
    }

    return (
        clean_sections,
        info,
    )


# ============================================================
# Render Clean Content
# ============================================================

def _render_clean_content(
    *,
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:

    output = []

    # ========================================================
    # Title
    # ========================================================

    title = (
        _normalize_text(
            title
        )
    )

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

    abstract = (
        _normalize_text(
            abstract
        )
    )

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

        heading = (
            _normalize_text(
                section.get(
                    "heading",
                    "Document Body",
                )
            )
        )

        level = (
            section.get(
                "level",
                2,
            )
        )

        try:

            level = int(
                level
            )

        except (
            TypeError,
            ValueError,
        ):

            level = 2

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

            block_type = str(
                block.get(
                    "type",
                    "paragraph",
                )
            )

            text = (
                _normalize_text(
                    block.get(
                        "text",
                        "",
                    )
                )
            )

            if not text:

                continue

            if (
                block_type
                == "list_item"
            ):

                output.append(
                    f"- {text}"
                )

            elif (
                block_type
                == "figure_caption"
            ):

                output.append(
                    (
                        "[FIGURE CAPTION] "
                        f"{text}"
                    )
                )

            elif (
                block_type
                == "equation"
            ):

                output.append(
                    (
                        "[EQUATION] "
                        f"{text}"
                    )
                )

            else:

                output.append(
                    text
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

def _word_count(
    text: str,
) -> int:

    return len(
        text.split()
    )


def _alpha_ratio(
    text: str,
) -> float:

    if not text:

        return 0.0

    characters = [
        char
        for char in text
        if not char.isspace()
    ]

    if not characters:

        return 0.0

    alphabetic = sum(
        1
        for char in characters
        if (
            "a"
            <= char.lower()
            <= "z"
        )
    )

    return round(
        alphabetic
        / len(
            characters
        ),
        4,
    )


def _percentage_reduction(
    before: int,
    after: int,
) -> float:

    if before <= 0:

        return 0.0

    reduction = (
        (
            before
            - after
        )
        / before
        * 100
    )

    return round(
        reduction,
        2,
    )


# ============================================================
# Content Hash
# ============================================================

def _content_hash(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Quality Gate
# ============================================================

def _evaluate_quality(
    clean_content: str,
    sections: list[dict],
) -> dict:

    char_count = len(
        clean_content
    )

    word_count = (
        _word_count(
            clean_content
        )
    )

    alpha_ratio = (
        _alpha_ratio(
            clean_content
        )
    )

    section_count = len(
        sections
    )

    reasons = []

    if (
        char_count
        < MIN_CHAR_COUNT
    ):

        reasons.append(
            (
                "char_count<"
                f"{MIN_CHAR_COUNT}"
            )
        )

    if (
        word_count
        < MIN_WORD_COUNT
    ):

        reasons.append(
            (
                "word_count<"
                f"{MIN_WORD_COUNT}"
            )
        )

    if (
        alpha_ratio
        < MIN_ALPHA_RATIO
    ):

        reasons.append(
            (
                "alpha_ratio<"
                f"{MIN_ALPHA_RATIO}"
            )
        )

    if (
        section_count
        < MIN_SECTION_COUNT
    ):

        reasons.append(
            (
                "section_count<"
                f"{MIN_SECTION_COUNT}"
            )
        )

    passed = (
        len(
            reasons
        )
        == 0
    )

    return {
        "passed": (
            passed
        ),
        "reasons": (
            reasons
        ),
        "char_count": (
            char_count
        ),
        "word_count": (
            word_count
        ),
        "alpha_ratio": (
            alpha_ratio
        ),
        "section_count": (
            section_count
        ),
    }


# ============================================================
# Clean One Document
# ============================================================

def clean_document(
    parsed_path: Path,
) -> dict:

    paper_dir = (
        parsed_path.parent
    )

    parsed = (
        _load_json(
            parsed_path
        )
    )

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    if not raw_path.exists():

        raise FileNotFoundError(
            f"raw_content.txt missing: "
            f"{raw_path}"
        )

    raw_content = (
        raw_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    raw_content = (
        _normalize_text(
            raw_content
        )
    )

    if not raw_content:

        raise RuntimeError(
            f"raw_content.txt empty: "
            f"{raw_path}"
        )

    title = (
        _normalize_text(
            parsed.get(
                "title",
                "",
            )
        )
    )

    abstract = (
        _normalize_text(
            parsed.get(
                "abstract",
                "",
            )
        )
    )

    sections = (
        parsed.get(
            "sections",
            [],
        )
    )

    if not isinstance(
        sections,
        list,
    ):

        raise RuntimeError(
            f"Invalid sections: "
            f"{parsed_path}"
        )

    # ========================================================
    # Clean Sections
    # ========================================================

    (
        clean_sections,
        cleaning_info,
    ) = (
        _clean_sections(
            sections
        )
    )

    # ========================================================
    # Render
    # ========================================================

    clean_content = (
        _render_clean_content(
            title=(
                title
            ),
            abstract=(
                abstract
            ),
            sections=(
                clean_sections
            ),
        )
    )

    # ========================================================
    # Quality
    # ========================================================

    quality = (
        _evaluate_quality(
            clean_content,
            clean_sections,
        )
    )

    digest = (
        _content_hash(
            clean_content
        )
        if clean_content
        else None
    )

    # ========================================================
    # Before / After
    # ========================================================

    before_chars = len(
        raw_content
    )

    after_chars = len(
        clean_content
    )

    before_words = (
        _word_count(
            raw_content
        )
    )

    after_words = (
        _word_count(
            clean_content
        )
    )

    statistics = {
        "before": {
            "char_count": (
                before_chars
            ),
            "word_count": (
                before_words
            ),
            "section_count": (
                len(
                    sections
                )
            ),
        },
        "after": {
            "char_count": (
                after_chars
            ),
            "word_count": (
                after_words
            ),
            "section_count": (
                len(
                    clean_sections
                )
            ),
            "alpha_ratio": (
                quality[
                    "alpha_ratio"
                ]
            ),
        },
        "reduction": {
            "char_percent": (
                _percentage_reduction(
                    before_chars,
                    after_chars,
                )
            ),
            "word_percent": (
                _percentage_reduction(
                    before_words,
                    after_words,
                )
            ),
        },
    }

    source_info = (
        parsed.get(
            "source_info",
            {},
        )
    )

    if not isinstance(
        source_info,
        dict,
    ):

        source_info = {}

    source_id = str(
        source_info.get(
            "source_id"
        )
        or paper_dir.name
    )

    topic_axis = str(
        source_info.get(
            "topic_axis"
        )
        or paper_dir.parent.name
    )

    result = {
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
        "abstract": (
            abstract
        ),
        "normalization_version": (
            NORMALIZATION_VERSION
        ),
        "sections": (
            clean_sections
        ),
        "clean_content": (
            clean_content
        ),
        "content_hash": (
            digest
        ),
        "quality": (
            quality
        ),
        "statistics": (
            statistics
        ),
        "cleaning": (
            cleaning_info
        ),
        "source_info": (
            source_info
        ),
    }

    return result


# ============================================================
# Save Clean Document
# ============================================================

def _save_cleaned_document(
    paper_dir: Path,
    cleaned: dict,
) -> None:

    clean_path = (
        paper_dir
        / "clean_content.txt"
    )

    json_path = (
        paper_dir
        / "cleaned_document.json"
    )

    clean_path.write_text(
        cleaned[
            "clean_content"
        ],
        encoding="utf-8",
    )

    _save_json(
        json_path,
        cleaned,
    )


# ============================================================
# Remove Stale Output After Failure
# ============================================================

def _remove_stale_outputs(
    paper_dir: Path,
) -> None:

    for filename in (
        "clean_content.txt",
        "cleaned_document.json",
    ):

        path = (
            paper_dir
            / filename
        )

        if not path.exists():

            continue

        try:

            path.unlink()

        except OSError:

            pass


# ============================================================
# Local Content Hash Duplicate Check
# ============================================================

def _mark_local_duplicates(
    results: list[dict],
) -> int:
    """
    Frozen 신규 NTRS 35편 내부의
    clean_content SHA-256 중복 검사.

    기존 DB arXiv/NTRS 문서와의 비교는
    DB Loader preflight에서 다시 수행한다.
    """

    seen: dict[
        str,
        dict,
    ] = {}

    duplicate_count = 0

    for result in results:

        if (
            result.get(
                "status"
            )
            != "success"
        ):

            continue

        result[
            "duplicate"
        ] = False

        content_hash = (
            result.get(
                "content_hash"
            )
        )

        if not content_hash:

            continue

        if (
            content_hash
            not in seen
        ):

            seen[
                content_hash
            ] = result

            continue

        duplicate_count += 1

        original = (
            seen[
                content_hash
            ]
        )

        result[
            "duplicate"
        ] = True

        result[
            "duplicate_of"
        ] = {
            "source_id": (
                original.get(
                    "source_id"
                )
            ),
            "topic_axis": (
                original.get(
                    "topic_axis"
                )
            ),
        }

    return duplicate_count


# ============================================================
# Report
# ============================================================

def _save_report(
    *,
    results: list[dict],
    duplicate_count: int,
    axis_success: dict[
        str,
        int,
    ],
    axis_quality_pass: dict[
        str,
        int,
    ],
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = sum(
        1
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    failed = (
        len(
            results
        )
        - success
    )

    quality_pass = sum(
        1
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
            and item.get(
                "quality_pass"
            )
        )
    )

    quality_fail = (
        success
        - quality_pass
    )

    words_before = sum(
        int(
            item.get(
                "words_before",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    words_after = sum(
        int(
            item.get(
                "words_after",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    chars_before = sum(
        int(
            item.get(
                "chars_before",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    chars_after = sum(
        int(
            item.get(
                "chars_after",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    payload = {
        "cleaner_version": (
            CLEANER_VERSION
        ),
        "source": (
            "ntrs"
        ),
        "selection_file": (
            str(
                SELECTION_FILE
            )
        ),
        "normalization_version": (
            NORMALIZATION_VERSION
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
            success
        ),
        "failed": (
            failed
        ),
        "quality_pass": (
            quality_pass
        ),
        "quality_fail": (
            quality_fail
        ),
        "duplicate_hashes": (
            duplicate_count
        ),
        "axis_success": (
            axis_success
        ),
        "axis_quality_pass": (
            axis_quality_pass
        ),
        "words_before": (
            words_before
        ),
        "words_after": (
            words_after
        ),
        "chars_before": (
            chars_before
        ),
        "chars_after": (
            chars_after
        ),
        "word_reduction_percent": (
            _percentage_reduction(
                words_before,
                words_after,
            )
        ),
        "char_reduction_percent": (
            _percentage_reduction(
                chars_before,
                chars_after,
            )
        ),
        "quality_thresholds": {
            "min_char_count": (
                MIN_CHAR_COUNT
            ),
            "min_word_count": (
                MIN_WORD_COUNT
            ),
            "min_alpha_ratio": (
                MIN_ALPHA_RATIO
            ),
            "min_section_count": (
                MIN_SECTION_COUNT
            ),
        },
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
        "Core-100 Cleaner"
    )

    print("=" * 78)

    print(
        f"Cleaner version : "
        f"{CLEANER_VERSION}"
    )

    print(
        f"Selection file  : "
        f"{SELECTION_FILE}"
    )

    print(
        f"Input root      : "
        f"{RESOLVED_ROOT}"
    )

    print(
        f"Report          : "
        f"{REPORT_FILE}"
    )

    print(
        f"Normalization   : "
        f"{NORMALIZATION_VERSION}"
    )

    print()

    print(
        "Quality thresholds:"
    )

    print(
        f"  chars    >= "
        f"{MIN_CHAR_COUNT}"
    )

    print(
        f"  words    >= "
        f"{MIN_WORD_COUNT}"
    )

    print(
        f"  alpha    >= "
        f"{MIN_ALPHA_RATIO}"
    )

    print(
        f"  sections >= "
        f"{MIN_SECTION_COUNT}"
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
    # Full preflight BEFORE mutation
    # ========================================================

    print()
    print(
        "[PREFLIGHT] Checking all "
        "35 parsed documents..."
    )

    selected_documents = (
        _build_selected_documents(
            selection
        )
    )

    print(
        f"[PREFLIGHT] PASS: "
        f"{len(selected_documents)} / "
        f"{EXPECTED_TOTAL}"
    )

    # ========================================================
    # Counters
    # ========================================================

    results = []

    axis_success = {
        topic_axis: 0
        for topic_axis
        in EXPECTED_COUNTS
    }

    axis_quality_pass = {
        topic_axis: 0
        for topic_axis
        in EXPECTED_COUNTS
    }

    # ========================================================
    # Clean Frozen 35 Only
    # ========================================================

    for (
        index,
        (
            topic_axis,
            source_id,
            parsed_path,
        ),
    ) in enumerate(
        selected_documents,
        start=1,
    ):

        paper_dir = (
            parsed_path.parent
        )

        print()
        print("-" * 78)

        print(
            f"[{index}/"
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

        print("-" * 78)

        try:

            cleaned = (
                clean_document(
                    parsed_path
                )
            )

            # =================================================
            # Identity must remain frozen
            # =================================================

            cleaned_source_id = str(
                cleaned.get(
                    "source_id",
                    "",
                )
            ).strip()

            cleaned_axis = str(
                cleaned.get(
                    "topic_axis",
                    "",
                )
            ).strip()

            if (
                cleaned_source_id
                != source_id
            ):

                raise RuntimeError(
                    "Cleaner source_id mismatch: "
                    f"{cleaned_source_id} "
                    f"!= {source_id}"
                )

            if (
                cleaned_axis
                != topic_axis
            ):

                raise RuntimeError(
                    "Cleaner topic_axis mismatch: "
                    f"{cleaned_axis} "
                    f"!= {topic_axis}"
                )

            # =================================================
            # Save
            # =================================================

            _save_cleaned_document(
                paper_dir,
                cleaned,
            )

            quality = (
                cleaned[
                    "quality"
                ]
            )

            stats = (
                cleaned[
                    "statistics"
                ]
            )

            axis_success[
                topic_axis
            ] += 1

            if quality[
                "passed"
            ]:

                axis_quality_pass[
                    topic_axis
                ] += 1

            # =================================================
            # Log
            # =================================================

            print(
                f"[OK] Title       : "
                f"{cleaned['title']}"
            )

            print(
                f"[OK] Words       : "
                f"{stats['before']['word_count']}"
                f" -> "
                f"{stats['after']['word_count']}"
            )

            print(
                f"[OK] Chars       : "
                f"{stats['before']['char_count']}"
                f" -> "
                f"{stats['after']['char_count']}"
            )

            print(
                f"[OK] Sections    : "
                f"{stats['before']['section_count']}"
                f" -> "
                f"{stats['after']['section_count']}"
            )

            print(
                f"[OK] Alpha ratio : "
                f"{quality['alpha_ratio']}"
            )

            print(
                f"[OK] Reduction   : "
                f"{stats['reduction']['word_percent']}%"
                f" words"
            )

            if quality[
                "passed"
            ]:

                print(
                    "[QUALITY] PASS"
                )

            else:

                print(
                    "[QUALITY] FAIL"
                )

                print(
                    f"[REASONS] "
                    f"{quality['reasons']}"
                )

            print(
                f"[HASH] "
                f"{cleaned['content_hash']}"
            )

            print(
                f"[SAVE] "
                f"{paper_dir / 'clean_content.txt'}"
            )

            print(
                f"[SAVE] "
                f"{paper_dir / 'cleaned_document.json'}"
            )

            results.append(
                {
                    "status": (
                        "success"
                    ),
                    "source_id": (
                        source_id
                    ),
                    "topic_axis": (
                        topic_axis
                    ),
                    "title": (
                        cleaned[
                            "title"
                        ]
                    ),
                    "quality_pass": (
                        quality[
                            "passed"
                        ]
                    ),
                    "quality_reasons": (
                        quality[
                            "reasons"
                        ]
                    ),
                    "content_hash": (
                        cleaned[
                            "content_hash"
                        ]
                    ),
                    "words_before": (
                        stats[
                            "before"
                        ][
                            "word_count"
                        ]
                    ),
                    "words_after": (
                        stats[
                            "after"
                        ][
                            "word_count"
                        ]
                    ),
                    "chars_before": (
                        stats[
                            "before"
                        ][
                            "char_count"
                        ]
                    ),
                    "chars_after": (
                        stats[
                            "after"
                        ][
                            "char_count"
                        ]
                    ),
                    "sections_before": (
                        stats[
                            "before"
                        ][
                            "section_count"
                        ]
                    ),
                    "sections_after": (
                        stats[
                            "after"
                        ][
                            "section_count"
                        ]
                    ),
                    "alpha_ratio": (
                        quality[
                            "alpha_ratio"
                        ]
                    ),
                    "references_removed": (
                        cleaned[
                            "cleaning"
                        ][
                            "references_removed"
                        ]
                    ),
                    "removed_blocks": (
                        cleaned[
                            "cleaning"
                        ][
                            "removed_blocks"
                        ]
                    ),
                    "duplicate": (
                        False
                    ),
                }
            )

        except Exception as exc:

            # stale output 방지
            _remove_stale_outputs(
                paper_dir
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
                    "source_id": (
                        source_id
                    ),
                    "topic_axis": (
                        topic_axis
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    # ========================================================
    # Local Duplicate QA
    # ========================================================

    duplicate_count = (
        _mark_local_duplicates(
            results
        )
    )

    # ========================================================
    # Summary Counts
    # ========================================================

    success = sum(
        1
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    failed = (
        len(
            results
        )
        - success
    )

    quality_pass = sum(
        1
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
            and item.get(
                "quality_pass"
            )
        )
    )

    quality_fail = (
        success
        - quality_pass
    )

    words_before = sum(
        int(
            item.get(
                "words_before",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    words_after = sum(
        int(
            item.get(
                "words_after",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    chars_before = sum(
        int(
            item.get(
                "chars_before",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    chars_after = sum(
        int(
            item.get(
                "chars_after",
                0,
            )
            or 0
        )
        for item
        in results
        if (
            item.get(
                "status"
            )
            == "success"
        )
    )

    word_reduction = (
        _percentage_reduction(
            words_before,
            words_after,
        )
    )

    char_reduction = (
        _percentage_reduction(
            chars_before,
            chars_after,
        )
    )

    # ========================================================
    # Report
    # ========================================================

    _save_report(
        results=(
            results
        ),
        duplicate_count=(
            duplicate_count
        ),
        axis_success=(
            axis_success
        ),
        axis_quality_pass=(
            axis_quality_pass
        ),
    )

    # ========================================================
    # Console Summary
    # ========================================================

    print()
    print("=" * 78)

    print(
        "NTRS CORE-100 "
        "CLEANING COMPLETED"
    )

    print("=" * 78)

    print(
        f"Selected       : "
        f"{len(results)}"
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
        f"Quality PASS   : "
        f"{quality_pass}"
    )

    print(
        f"Quality FAIL   : "
        f"{quality_fail}"
    )

    print(
        f"Duplicates     : "
        f"{duplicate_count}"
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
            f"{expected_count} "
            f"(quality "
            f"{axis_quality_pass[topic_axis]})"
        )

    print()

    print(
        f"Words before   : "
        f"{words_before}"
    )

    print(
        f"Words after    : "
        f"{words_after}"
    )

    print(
        f"Word reduction : "
        f"{word_reduction}%"
    )

    print()

    print(
        f"Chars before   : "
        f"{chars_before}"
    )

    print(
        f"Chars after    : "
        f"{chars_after}"
    )

    print(
        f"Char reduction : "
        f"{char_reduction}%"
    )

    print()

    print(
        f"Report         : "
        f"{REPORT_FILE}"
    )

    print("=" * 78)

    # ========================================================
    # Hard Gate
    # ========================================================

    all_axis_success = all(
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

    all_axis_quality = all(
        axis_quality_pass[
            topic_axis
        ]
        == expected_count
        for (
            topic_axis,
            expected_count,
        )
        in EXPECTED_COUNTS.items()
    )

    if (
        len(
            results
        )
        == EXPECTED_TOTAL
        and success
        == EXPECTED_TOTAL
        and failed
        == 0
        and quality_pass
        == EXPECTED_TOTAL
        and quality_fail
        == 0
        and duplicate_count
        == 0
        and all_axis_success
        and all_axis_quality
    ):

        print(
            "[PASS] All 35 selected "
            "NTRS papers passed cleaning."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Run the NTRS Core-100 "
            "DB Loader."
        )

        print("=" * 78)

        return

    # ========================================================
    # Failed Gate
    # ========================================================

    print(
        "[CHECK] Cleaner Gate failed."
    )

    print()

    if quality_fail > 0:

        print(
            "Quality failures:"
        )

        for item in results:

            if (
                item.get(
                    "status"
                )
                == "success"
                and not item.get(
                    "quality_pass"
                )
            ):

                print(
                    f"  - "
                    f"{item['topic_axis']} / "
                    f"{item['source_id']}"
                )

                print(
                    f"    reasons: "
                    f"{item.get('quality_reasons')}"
                )

    if duplicate_count > 0:

        print()

        print(
            "Duplicate hashes:"
        )

        for item in results:

            if item.get(
                "duplicate"
            ):

                print(
                    f"  - "
                    f"{item.get('source_id')} "
                    f"duplicates "
                    f"{item.get('duplicate_of')}"
                )

    print()
    print(
        "Do NOT run the DB Loader yet."
    )

    print(
        "Repair only the failed "
        "document(s), then rerun Cleaner."
    )

    print("=" * 78)

    raise RuntimeError(
        "NTRS Core-100 Cleaner "
        f"Gate failed: "
        f"quality "
        f"{quality_pass}/"
        f"{EXPECTED_TOTAL}, "
        f"duplicates="
        f"{duplicate_count}."
    )


if __name__ == "__main__":
    main()