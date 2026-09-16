import hashlib
import json
import re
import unicodedata
from pathlib import Path

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - arXiv Core-100 Cleaner
# ============================================================
#
# 대상:
#
# Human QA를 통해 선택된 신규 arXiv 35편만 처리한다.
#
#   rover_autonomy       12
#   onboard_ai           11
#   satellite_autonomy   12
#   -----------------------
#   TOTAL                35
#
#
# 입력:
#
# data/selections/
#     arxiv_core100_selected.json
#
# data/tmp/arxiv_resolved/
#     <topic_axis>/
#         <source_id>/
#             parsed_document.json
#
#
# 출력:
#
#             clean_content.txt
#             cleaned_document.json
#
#
# 처리:
#
# parsed_document
#       ↓
# References 제거
# Acknowledgments/Funding 제거
# Noise normalization
#       ↓
# clean_content
#       ↓
# Quality Gate
#       ↓
# SHA-256 content hash
#       ↓
# duplicate check
#
# ============================================================


CLEANER_VERSION = "core100_v1"


# ============================================================
# Paths
# ============================================================

RESOLVED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tmp"
    / "arxiv_resolved"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "selections"
    / "arxiv_core100_selected.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "arxiv_core100_cleaning_report.json"
)


# ============================================================
# Expected Counts
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
# Quality Thresholds
# ============================================================
#
# 기존 파일럿 threshold 유지.
#
# 기준을 낮춰서 억지로 PASS시키지 않는다.
#
# 짧은 PDF/TeX가 FAIL하면:
#
#   1. 실제 parsing 품질 확인
#   2. 원 논문 자체가 짧은지 확인
#   3. 필요하면 review pool에서 다른 논문으로 교체
#
# ============================================================

MIN_CHAR_COUNT = 5000
MIN_WORD_COUNT = 800
MIN_ALPHA_RATIO = 0.45
MIN_SECTION_COUNT = 2


# ============================================================
# Section Rules
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
    "funding statement",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "author contributions",
}


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
# Basic Normalization
# ============================================================

def _normalize_unicode(
    text: str,
) -> str:

    return unicodedata.normalize(
        "NFC",
        text,
    )


def _remove_null_bytes(
    text: str,
) -> str:

    return text.replace(
        "\x00",
        "",
    )


def _normalize_spaces(
    text: str,
) -> str:

    text = text.replace(
        "\u00a0",
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

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def _normalize_text(
    text: str,
) -> str:

    if text is None:
        return ""

    text = str(
        text
    )

    text = _normalize_unicode(
        text
    )

    text = _remove_null_bytes(
        text
    )

    text = _normalize_spaces(
        text
    )

    return text.strip()


# ============================================================
# Heading Helpers
# ============================================================

def _normalize_heading_key(
    heading: str,
) -> str:

    heading = _normalize_text(
        heading
    ).lower()

    # Markdown residue
    heading = re.sub(
        r"^#+\s*",
        "",
        heading,
    )

    # --------------------------------------------------------
    # Section number 제거
    #
    # 1 Introduction
    # 2. Methods
    # IV. Results
    # A. Related Work
    # --------------------------------------------------------

    heading = re.sub(
        r"^\s*(?:"
        r"\d+(?:\.\d+)*"
        r"|[ivxlcdm]+"
        r"|[a-z]"
        r")"
        r"[\.\):\-]?\s+",
        "",
        heading,
        flags=re.IGNORECASE,
    )

    heading = heading.strip(
        " .:-"
    )

    return heading


def _is_reference_heading(
    heading: str,
) -> bool:

    key = _normalize_heading_key(
        heading
    )

    if key in REFERENCE_HEADINGS:
        return True

    if key.startswith(
        "references "
    ):
        return True

    if key.startswith(
        "bibliography "
    ):
        return True

    return False


def _should_remove_section(
    heading: str,
) -> bool:

    key = _normalize_heading_key(
        heading
    )

    return (
        key
        in REMOVE_SECTION_HEADINGS
    )


# ============================================================
# Inline Residue
# ============================================================

def _remove_inline_residue(
    text: str,
) -> str:

    text = _normalize_text(
        text
    )

    # --------------------------------------------------------
    # Permalink residue
    # --------------------------------------------------------

    text = text.replace(
        "¶",
        "",
    )

    # --------------------------------------------------------
    # IEEE copyright residue
    # --------------------------------------------------------

    text = re.sub(
        r"\b\d{3,4}-\d{3,4}-\d{3,4}-\d/"
        r"\d{2}/\$[\d.]+\s*©\s*\d{4}\s*IEEE\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"©\s*\d{4}\s*IEEE\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Project page residue
    # --------------------------------------------------------

    text = re.sub(
        r"\bProject page\s*:\s*\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Thanks label residue
    # --------------------------------------------------------

    text = re.sub(
        r"\bThanks\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Figure / Table Caption
# ============================================================

def _clean_figure_caption(
    text: str,
) -> str:

    text = _remove_inline_residue(
        text
    )

    text = re.sub(
        r"^\s*(?:"
        r"figure"
        r"|fig\."
        r"|fig"
        r")\s*"
        r"[A-Za-z0-9.\-()]+"
        r"\s*[:.\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _clean_table_caption(
    text: str,
) -> str:

    text = _remove_inline_residue(
        text
    )

    text = re.sub(
        r"^\s*table\s*"
        r"[A-Za-z0-9.\-()]+"
        r"\s*[:.\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ============================================================
# Equation
# ============================================================

def _clean_equation(
    text: str,
) -> str:

    return _normalize_text(
        text
    )


# ============================================================
# Block Cleaning
# ============================================================

def _clean_block(
    block: dict,
) -> dict | None:

    block_type = str(
        block.get(
            "type",
            "paragraph",
        )
        or "paragraph"
    )

    text = str(
        block.get(
            "text",
            "",
        )
        or ""
    )

    if not text:
        return None

    # HTML parser bibliography marker
    if block.get(
        "in_bibliography",
        False,
    ):

        return None

    # ========================================================
    # Figure Caption
    # ========================================================

    if (
        block_type
        == "figure_caption"
    ):

        text = (
            _clean_figure_caption(
                text
            )
        )

        if len(text) < 20:
            return None

    # ========================================================
    # Table Caption
    # ========================================================

    elif (
        block_type
        == "table_caption"
    ):

        text = (
            _clean_table_caption(
                text
            )
        )

        if len(text) < 20:
            return None

    # ========================================================
    # Equation
    # ========================================================

    elif (
        block_type
        == "equation"
    ):

        text = (
            _clean_equation(
                text
            )
        )

        if not text:
            return None

    # ========================================================
    # Paragraph / List / Other
    # ========================================================

    else:

        text = (
            _remove_inline_residue(
                text
            )
        )

        if len(text) < 2:
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
# Section Cleaning
# ============================================================

def _clean_sections(
    sections: list[dict],
) -> tuple[
    list[dict],
    dict,
]:

    cleaned_sections: list[dict] = []

    removed_reference_sections = 0
    removed_other_sections = 0
    removed_blocks = 0

    for section in sections:

        heading = str(
            section.get(
                "heading",
                "Document Body",
            )
            or "Document Body"
        )

        heading = (
            _normalize_text(
                heading
            )
        )

        # ====================================================
        # References / Bibliography
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

            removed_reference_sections += 1
            continue

        # ====================================================
        # Back Matter
        # ====================================================

        if (
            _should_remove_section(
                heading
            )
        ):

            removed_other_sections += 1
            continue

        original_blocks = (
            section.get(
                "blocks",
                [],
            )
        )

        if not isinstance(
            original_blocks,
            list,
        ):

            continue

        cleaned_blocks: list[dict] = []

        previous_key = None

        for block in original_blocks:

            if not isinstance(
                block,
                dict,
            ):

                removed_blocks += 1
                continue

            cleaned_block = (
                _clean_block(
                    block
                )
            )

            if (
                cleaned_block
                is None
            ):

                removed_blocks += 1
                continue

            # ------------------------------------------------
            # 연속 exact duplicate 제거
            # ------------------------------------------------

            current_key = (
                cleaned_block[
                    "type"
                ],
                cleaned_block[
                    "text"
                ],
            )

            if (
                previous_key
                == current_key
            ):

                removed_blocks += 1
                continue

            previous_key = (
                current_key
            )

            cleaned_blocks.append(
                cleaned_block
            )

        if not cleaned_blocks:
            continue

        cleaned_sections.append(
            {
                "heading": (
                    heading
                ),

                "level": (
                    section.get(
                        "level",
                        2,
                    )
                ),

                "blocks": (
                    cleaned_blocks
                ),
            }
        )

    stats = {
        "removed_reference_sections": (
            removed_reference_sections
        ),

        "removed_other_sections": (
            removed_other_sections
        ),

        "removed_blocks": (
            removed_blocks
        ),
    }

    return (
        cleaned_sections,
        stats,
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
                "",
            )
            or ""
        ).strip()

        level = section.get(
            "level",
            2,
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

        if heading:

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

            text = str(
                block.get(
                    "text",
                    "",
                )
            ).strip()

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
                    f"[FIGURE CAPTION] "
                    f"{text}"
                )

            elif (
                block_type
                == "table_caption"
            ):

                output.append(
                    f"[TABLE CAPTION] "
                    f"{text}"
                )

            elif (
                block_type
                == "equation"
            ):

                output.append(
                    f"[EQUATION] "
                    f"{text}"
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

    result = result.replace(
        "\r\n",
        "\n",
    )

    result = result.replace(
        "\r",
        "\n",
    )

    result = re.sub(
        r"[ \t]+\n",
        "\n",
        result,
    )

    result = re.sub(
        r"\n{3,}",
        "\n\n",
        result,
    )

    result = (
        _normalize_unicode(
            result
        )
    )

    result = (
        _remove_null_bytes(
            result
        )
    )

    return result.strip()


# ============================================================
# Content Hash
# ============================================================

def _content_hash(
    clean_content: str,
) -> str:

    return hashlib.sha256(
        clean_content.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Quality Metrics
# ============================================================

def _alpha_ratio(
    text: str,
) -> float:

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

    return (
        alphabetic
        / len(characters)
    )


def _quality_check(
    clean_content: str,
    sections: list[dict],
) -> dict:

    char_count = len(
        clean_content
    )

    word_count = len(
        clean_content.split()
    )

    section_count = len(
        sections
    )

    alpha_ratio = (
        _alpha_ratio(
            clean_content
        )
    )

    reasons: list[str] = []

    if (
        char_count
        < MIN_CHAR_COUNT
    ):

        reasons.append(
            f"char_count<{MIN_CHAR_COUNT}"
        )

    if (
        word_count
        < MIN_WORD_COUNT
    ):

        reasons.append(
            f"word_count<{MIN_WORD_COUNT}"
        )

    if (
        alpha_ratio
        < MIN_ALPHA_RATIO
    ):

        reasons.append(
            f"alpha_ratio<{MIN_ALPHA_RATIO}"
        )

    if (
        section_count
        < MIN_SECTION_COUNT
    ):

        reasons.append(
            f"section_count<{MIN_SECTION_COUNT}"
        )

    return {
        "passed": (
            len(reasons)
            == 0
        ),

        "char_count": (
            char_count
        ),

        "word_count": (
            word_count
        ),

        "section_count": (
            section_count
        ),

        "alpha_ratio": round(
            alpha_ratio,
            4,
        ),

        "reasons": (
            reasons
        ),
    }


# ============================================================
# Selection Manifest
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
            "Selection JSON does not contain "
            "'selected_documents'."
        )

    validated = {}

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
                "selection is not a list."
            )

        normalized_ids = []

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
                    "Duplicate selected "
                    f"source_id: {source_id}"
                )

            seen_ids.add(
                source_id
            )

            normalized_ids.append(
                source_id
            )

        if (
            len(normalized_ids)
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count}, "
                f"found "
                f"{len(normalized_ids)}"
            )

        validated[
            topic_axis
        ] = normalized_ids

    total = sum(
        len(items)
        for items
        in validated.values()
    )

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL}, "
            f"found {total}."
        )

    return validated


# ============================================================
# Find Selected Parsed Documents
# ============================================================

def _find_selected_parsed_documents(
    selection: dict[
        str,
        list[str],
    ],
) -> list[Path]:

    parsed_files = []

    errors = []

    for (
        topic_axis,
        source_ids,
    ) in selection.items():

        for source_id in source_ids:

            parsed_path = (
                RESOLVED_ROOT
                / topic_axis
                / source_id
                / "parsed_document.json"
            )

            if not parsed_path.exists():

                errors.append(
                    f"{topic_axis} / "
                    f"{source_id}"
                )

                continue

            parsed_files.append(
                parsed_path
            )

    if errors:

        print()

        print(
            "Missing parsed documents:"
        )

        for item in errors:

            print(
                f"  - {item}"
            )

        raise RuntimeError(
            f"{len(errors)} selected "
            "parsed_document.json "
            "file(s) missing."
        )

    if (
        len(parsed_files)
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL} parsed "
            f"documents, found "
            f"{len(parsed_files)}."
        )

    return parsed_files


# ============================================================
# Clean One Document
# ============================================================

def clean_document(
    parsed_path: Path,
) -> dict:

    parsed = (
        _load_json(
            parsed_path
        )
    )

    title = (
        _remove_inline_residue(
            parsed.get(
                "title",
                "",
            )
        )
    )

    abstract = (
        _remove_inline_residue(
            parsed.get(
                "abstract",
                "",
            )
        )
    )

    original_sections = (
        parsed.get(
            "sections",
            [],
        )
    )

    if not isinstance(
        original_sections,
        list,
    ):

        raise RuntimeError(
            "parsed_document sections "
            "is not a list."
        )

    (
        cleaned_sections,
        cleaning_stats,
    ) = _clean_sections(
        original_sections
    )

    clean_content = (
        _render_clean_content(
            title=title,
            abstract=abstract,
            sections=cleaned_sections,
        )
    )

    quality = (
        _quality_check(
            clean_content,
            cleaned_sections,
        )
    )

    content_hash = (
        _content_hash(
            clean_content
        )
    )

    raw_stats = (
        parsed.get(
            "stats",
            {},
        )
    )

    before_chars = int(
        raw_stats.get(
            "char_count",
            0,
        )
        or 0
    )

    before_words = int(
        raw_stats.get(
            "word_count",
            0,
        )
        or 0
    )

    after_chars = int(
        quality[
            "char_count"
        ]
    )

    after_words = int(
        quality[
            "word_count"
        ]
    )

    char_reduction_ratio = 0.0
    word_reduction_ratio = 0.0

    if before_chars > 0:

        char_reduction_ratio = (
            1
            - (
                after_chars
                / before_chars
            )
        )

    if before_words > 0:

        word_reduction_ratio = (
            1
            - (
                after_words
                / before_words
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
            cleaned_sections
        ),

        "clean_content": (
            clean_content
        ),

        "content_hash": (
            content_hash
        ),

        "normalization_version": (
            settings.normalization_version
        ),

        "quality": (
            quality
        ),

        "cleaning_stats": {
            **cleaning_stats,

            "before_char_count": (
                before_chars
            ),

            "after_char_count": (
                after_chars
            ),

            "before_word_count": (
                before_words
            ),

            "after_word_count": (
                after_words
            ),

            "char_reduction_ratio": round(
                char_reduction_ratio,
                4,
            ),

            "word_reduction_ratio": round(
                word_reduction_ratio,
                4,
            ),
        },
    }


# ============================================================
# Save One Cleaned Document
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

    payload = {
        "title": (
            cleaned[
                "title"
            ]
        ),

        "abstract": (
            cleaned[
                "abstract"
            ]
        ),

        "sections": (
            cleaned[
                "sections"
            ]
        ),

        "content_hash": (
            cleaned[
                "content_hash"
            ]
        ),

        "normalization_version": (
            cleaned[
                "normalization_version"
            ]
        ),

        "quality": (
            cleaned[
                "quality"
            ]
        ),

        "cleaning_stats": (
            cleaned[
                "cleaning_stats"
            ]
        ),
    }

    _save_json(
        json_path,
        payload,
    )


# ============================================================
# Duplicate Check
# ============================================================

def _check_duplicate_hashes(
    results: list[dict],
) -> None:

    seen: dict[
        str,
        str,
    ] = {}

    for result in results:

        if (
            result.get(
                "status"
            )
            != "success"
        ):
            continue

        content_hash = (
            result.get(
                "content_hash"
            )
        )

        source_id = (
            result.get(
                "source_id"
            )
        )

        if not content_hash:
            continue

        if content_hash in seen:

            result[
                "duplicate_of"
            ] = seen[
                content_hash
            ]

        else:

            seen[
                content_hash
            ] = source_id


# ============================================================
# Save Report
# ============================================================

def _save_report(
    results: list[dict],
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _check_duplicate_hashes(
        results
    )

    success_count = sum(
        1
        for result in results
        if (
            result.get(
                "status"
            )
            == "success"
        )
    )

    failed_count = (
        len(results)
        - success_count
    )

    quality_pass_count = sum(
        1
        for result in results
        if (
            result.get(
                "status"
            )
            == "success"
            and result.get(
                "quality",
                {},
            ).get(
                "passed",
                False,
            )
        )
    )

    quality_fail_count = (
        success_count
        - quality_pass_count
    )

    duplicate_count = sum(
        1
        for result in results
        if result.get(
            "duplicate_of"
        )
    )

    total_before_words = sum(
        result.get(
            "cleaning_stats",
            {},
        ).get(
            "before_word_count",
            0,
        )
        for result in results
        if (
            result.get(
                "status"
            )
            == "success"
        )
    )

    total_after_words = sum(
        result.get(
            "cleaning_stats",
            {},
        ).get(
            "after_word_count",
            0,
        )
        for result in results
        if (
            result.get(
                "status"
            )
            == "success"
        )
    )

    payload = {
        "cleaner_version": (
            CLEANER_VERSION
        ),

        "selection_file": (
            str(
                SELECTION_FILE
            )
        ),

        "total": (
            len(results)
        ),

        "success": (
            success_count
        ),

        "failed": (
            failed_count
        ),

        "quality_pass": (
            quality_pass_count
        ),

        "quality_fail": (
            quality_fail_count
        ),

        "duplicate_hashes": (
            duplicate_count
        ),

        "total_before_words": (
            total_before_words
        ),

        "total_after_words": (
            total_after_words
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
    print("=" * 70)

    print(
        "TEAM B - arXiv Core-100 Cleaner"
    )

    print("=" * 70)

    print(
        f"Version        : "
        f"{CLEANER_VERSION}"
    )

    print(
        f"Input root     : "
        f"{RESOLVED_ROOT}"
    )

    print(
        f"Selection file : "
        f"{SELECTION_FILE}"
    )

    print(
        f"Normalization  : "
        f"{settings.normalization_version}"
    )

    print()

    print(
        "Quality thresholds:"
    )

    print(
        f"  min chars    : "
        f"{MIN_CHAR_COUNT}"
    )

    print(
        f"  min words    : "
        f"{MIN_WORD_COUNT}"
    )

    print(
        f"  alpha ratio  : "
        f"{MIN_ALPHA_RATIO}"
    )

    print(
        f"  min sections : "
        f"{MIN_SECTION_COUNT}"
    )

    # ========================================================
    # Selection / Precheck
    # ========================================================

    try:

        selection = (
            _load_selection()
        )

        parsed_files = (
            _find_selected_parsed_documents(
                selection
            )
        )

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            "CLEANER PRECHECK FAILED"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("=" * 70)

        return

    print()
    print(
        "Selected documents:"
    )

    for (
        topic_axis,
        source_ids,
    ) in selection.items():

        print(
            f"  {topic_axis:22} : "
            f"{len(source_ids)}"
        )

    print(
        f"  {'TOTAL':22} : "
        f"{len(parsed_files)}"
    )

    results: list[dict] = []

    # ========================================================
    # Clean Selected 35
    # ========================================================

    for (
        index,
        parsed_path,
    ) in enumerate(
        parsed_files,
        start=1,
    ):

        paper_dir = (
            parsed_path.parent
        )

        topic_axis = (
            paper_dir
            .parent
            .name
        )

        source_id = (
            paper_dir.name
        )

        print()
        print("-" * 70)

        print(
            f"[{index}/"
            f"{len(parsed_files)}]"
        )

        print(
            f"[AXIS] "
            f"{topic_axis}"
        )

        print(
            f"[ID]   "
            f"{source_id}"
        )

        print("-" * 70)

        try:

            cleaned = (
                clean_document(
                    parsed_path
                )
            )

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
                    "cleaning_stats"
                ]
            )

            quality_label = (
                "PASS"
                if quality[
                    "passed"
                ]
                else "FAIL"
            )

            print(
                f"[OK] Title       : "
                f"{cleaned['title']}"
            )

            print(
                f"[OK] Words       : "
                f"{stats['before_word_count']} "
                f"-> "
                f"{stats['after_word_count']}"
            )

            print(
                f"[OK] Chars       : "
                f"{stats['before_char_count']} "
                f"-> "
                f"{stats['after_char_count']}"
            )

            print(
                f"[OK] Sections    : "
                f"{quality['section_count']}"
            )

            print(
                f"[OK] Alpha ratio : "
                f"{quality['alpha_ratio']}"
            )

            print(
                f"[QUALITY] "
                f"{quality_label}"
            )

            if not quality[
                "passed"
            ]:

                print(
                    "[QUALITY] Reasons: "
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
                        cleaned[
                            "title"
                        ]
                    ),

                    "content_hash": (
                        cleaned[
                            "content_hash"
                        ]
                    ),

                    "normalization_version": (
                        cleaned[
                            "normalization_version"
                        ]
                    ),

                    "quality": (
                        quality
                    ),

                    "cleaning_stats": (
                        stats
                    ),

                    "clean_content_path": str(
                        paper_dir
                        / "clean_content.txt"
                    ),

                    "cleaned_json_path": str(
                        paper_dir
                        / "cleaned_document.json"
                    ),
                }
            )

        except Exception as exc:

            print(
                f"[ERROR] "
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

        # 중간 중단되어도 진행 report 보존
        _save_report(
            results
        )

    # ========================================================
    # Final Duplicate Check / Report
    # ========================================================

    _check_duplicate_hashes(
        results
    )

    _save_report(
        results
    )

    # ========================================================
    # Summary
    # ========================================================

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
        len(results)
        - success_count
    )

    quality_pass = sum(
        1
        for item in results
        if (
            item.get(
                "status"
            )
            == "success"
            and item.get(
                "quality",
                {},
            ).get(
                "passed",
                False,
            )
        )
    )

    quality_fail = (
        success_count
        - quality_pass
    )

    duplicate_count = sum(
        1
        for item in results
        if item.get(
            "duplicate_of"
        )
    )

    before_words = sum(
        item.get(
            "cleaning_stats",
            {},
        ).get(
            "before_word_count",
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

    after_words = sum(
        item.get(
            "cleaning_stats",
            {},
        ).get(
            "after_word_count",
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

    print()
    print("=" * 70)

    print(
        "ARXIV CORE-100 "
        "CLEANING COMPLETED"
    )

    print("=" * 70)

    print(
        f"Selected     : "
        f"{len(results)}"
    )

    print(
        f"Success      : "
        f"{success_count}"
    )

    print(
        f"Failed       : "
        f"{failed_count}"
    )

    print(
        f"Quality PASS : "
        f"{quality_pass}"
    )

    print(
        f"Quality FAIL : "
        f"{quality_fail}"
    )

    print(
        f"Duplicates   : "
        f"{duplicate_count}"
    )

    print(
        f"Words before : "
        f"{before_words}"
    )

    print(
        f"Words after  : "
        f"{after_words}"
    )

    if before_words > 0:

        reduction = (
            1
            - (
                after_words
                / before_words
            )
        )

        print(
            f"Reduction    : "
            f"{reduction:.2%}"
        )

    print(
        f"Report       : "
        f"{REPORT_FILE}"
    )

    print("=" * 70)

    # ========================================================
    # Final Gate
    # ========================================================

    if failed_count > 0:

        print(
            "[FAIL] Cleaner execution "
            "errors exist."
        )

        print(
            "Do NOT run DB Loader yet."
        )

    elif duplicate_count > 0:

        print(
            "[CHECK] Duplicate content "
            "hash detected."
        )

        print(
            "Do NOT run DB Loader yet."
        )

    elif quality_fail > 0:

        print(
            "[CHECK] Some selected papers "
            "failed the Quality Gate."
        )

        print(
            "Inspect those papers before "
            "running DB Loader."
        )

    elif (
        quality_pass
        == EXPECTED_TOTAL
    ):

        print(
            "[PASS] All 35 selected "
            "arXiv papers passed cleaning."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "Run the Core-100 "
            "arXiv DB Loader."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()