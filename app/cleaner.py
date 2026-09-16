import hashlib
import json
import re
import unicodedata
from pathlib import Path

from app.config import PROJECT_ROOT, settings


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
    / "arxiv_cleaning_report.json"
)


# ============================================================
# Quality Thresholds
# ============================================================
#
# 파일럿 기준.
#
# 지금은 너무 공격적으로 필터링하지 않는다.
# Core를 100 -> 300 -> 1,000+으로 확대하면서
# 실제 통계를 보고 조정한다.
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
# Basic Normalization
# ============================================================

def _normalize_unicode(
    text: str,
) -> str:
    """
    Unicode NFC 정규화.

    같은 글자가 서로 다른 Unicode 조합으로
    저장되는 문제를 줄인다.
    """

    return unicodedata.normalize(
        "NFC",
        text,
    )


def _remove_null_bytes(
    text: str,
) -> str:
    """
    PostgreSQL text column에서 문제가 될 수 있는
    NULL 문자 제거.
    """

    return text.replace(
        "\x00",
        "",
    )


def _normalize_spaces(
    text: str,
) -> str:
    """
    일반적인 공백 정리.

    줄바꿈 구조 자체는 없애지 않는다.
    """

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
    """
    기본 문자열 정규화.
    """

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
    """
    section heading 비교용 문자열.

    예:
    "VI. REFERENCES"
    ->
    "references"
    """

    heading = _normalize_text(
        heading
    ).lower()

    # Markdown heading residue
    heading = re.sub(
        r"^#+\s*",
        "",
        heading,
    )

    # section 번호 제거
    #
    # 1 Introduction
    # 2. Methods
    # IV. Results
    # A. Related Work
    #
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
    """
    References/Bibliography section인지 판단.
    """

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
    """
    학습/RAG에 크게 필요하지 않은
    back-matter section 제거.

    공격적으로 제거하지 않는다.
    """

    key = _normalize_heading_key(
        heading
    )

    return key in REMOVE_SECTION_HEADINGS


# ============================================================
# Residue Removal
# ============================================================

def _remove_inline_residue(
    text: str,
) -> str:
    """
    HTML 변환 과정에서 남을 수 있는
    최소한의 noise만 제거.

    논문 의미를 훼손할 정도의
    aggressive cleaning은 하지 않는다.
    """

    text = _normalize_text(
        text
    )

    # --------------------------------------------------------
    # Permalink / UI residue
    # --------------------------------------------------------

    text = text.replace(
        "¶",
        "",
    )

    # --------------------------------------------------------
    # 매우 명백한 IEEE copyright residue
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
    # Project-page residue
    # --------------------------------------------------------

    text = re.sub(
        r"\bProject page\s*:\s*\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Thanks label residue
    #
    # 주의:
    # 문장 전체를 자르지 않고 label만 정리한다.
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
# Figure Caption Cleaning
# ============================================================

def _clean_figure_caption(
    text: str,
) -> str:
    """
    Figure caption은 완전히 버리지 않는다.

    scientific RAG에서 figure caption은
    실험 조건/결과 설명을 포함하는 경우가 많기 때문이다.

    대신 Figure 1:, Fig. 2 등의 label만 정리한다.
    """

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


# ============================================================
# Equation Cleaning
# ============================================================

def _clean_equation(
    text: str,
) -> str:
    """
    수식은 삭제하지 않는다.

    너무 aggressive한 수식 정제는
    논문의 의미를 망칠 수 있으므로
    공백/Unicode만 정리한다.
    """

    return _normalize_text(
        text
    )


# ============================================================
# Block Cleaning
# ============================================================

def _clean_block(
    block: dict,
) -> dict | None:
    """
    parser block 하나 정제.

    반환:
    cleaned block 또는 None
    """

    block_type = block.get(
        "type",
        "paragraph",
    )

    text = block.get(
        "text",
        "",
    )

    if not text:
        return None

    # References 내부 block은 제거
    if block.get(
        "in_bibliography",
        False,
    ):
        return None

    # --------------------------------------------------------
    # Figure caption
    # --------------------------------------------------------

    if block_type == "figure_caption":

        text = _clean_figure_caption(
            text
        )

        # 너무 짧은 caption은 정보량이 거의 없음
        if len(text) < 20:
            return None

    # --------------------------------------------------------
    # Equation
    # --------------------------------------------------------

    elif block_type == "equation":

        text = _clean_equation(
            text
        )

        if not text:
            return None

    # --------------------------------------------------------
    # Paragraph / List
    # --------------------------------------------------------

    else:

        text = _remove_inline_residue(
            text
        )

        if len(text) < 2:
            return None

    cleaned = {
        "type": block_type,
        "text": text,
    }

    return cleaned


# ============================================================
# Section Cleaning
# ============================================================

def _clean_sections(
    sections: list[dict],
) -> tuple[list[dict], dict]:
    """
    전체 section 정제.

    Returns
    -------
    cleaned_sections
    section_stats
    """

    cleaned_sections: list[dict] = []

    removed_reference_sections = 0
    removed_other_sections = 0
    removed_blocks = 0

    for section in sections:

        heading = section.get(
            "heading",
            "Document Body",
        )

        heading = _normalize_text(
            heading
        )

        # ----------------------------------------------------
        # References 제거
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Acknowledgments 등 제거
        # ----------------------------------------------------

        if _should_remove_section(
            heading
        ):

            removed_other_sections += 1

            continue

        original_blocks = section.get(
            "blocks",
            [],
        )

        cleaned_blocks: list[dict] = []

        previous_key = None

        for block in original_blocks:

            cleaned_block = _clean_block(
                block
            )

            if cleaned_block is None:

                removed_blocks += 1
                continue

            # ------------------------------------------------
            # 연속 exact duplicate 제거
            #
            # HTML 변환기 때문에 같은 paragraph/caption이
            # 바로 반복되는 경우만 제거한다.
            #
            # 문서 전체 global dedup은 하지 않는다.
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

            previous_key = current_key

            cleaned_blocks.append(
                cleaned_block
            )

        if not cleaned_blocks:
            continue

        cleaned_sections.append(
            {
                "heading": heading,
                "level": section.get(
                    "level",
                    2,
                ),
                "blocks": cleaned_blocks,
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
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:
    """
    clean_content 생성.

    section heading 구조는 유지한다.
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

        heading = section[
            "heading"
        ]

        level = section.get(
            "level",
            2,
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

        for block in section[
            "blocks"
        ]:

            block_type = block[
                "type"
            ]

            text = block[
                "text"
            ]

            if block_type == "list_item":

                output.append(
                    f"- {text}"
                )

            elif block_type == "figure_caption":

                output.append(
                    f"[FIGURE CAPTION] {text}"
                )

            elif block_type == "equation":

                output.append(
                    f"[EQUATION] {text}"
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

    # --------------------------------------------------------
    # whitespace normalization
    # --------------------------------------------------------

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

    result = _normalize_unicode(
        result
    )

    result = _remove_null_bytes(
        result
    )

    return result.strip()


# ============================================================
# Content Hash
# ============================================================

def _content_hash(
    clean_content: str,
) -> str:
    """
    SHA-256 content hash.

    DB dedup에 사용한다.
    """

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
    """
    영어 alphabet 비율.

    whitespace를 제외한 문자 중
    A-Z / a-z가 차지하는 비율.
    """

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
            "a" <= char.lower() <= "z"
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
    """
    Core corpus 최소 품질 검사.

    지금은 파일럿용 baseline.
    """

    char_count = len(
        clean_content
    )

    word_count = len(
        clean_content.split()
    )

    section_count = len(
        sections
    )

    alpha_ratio = _alpha_ratio(
        clean_content
    )

    reasons = []

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

    passed = (
        len(reasons)
        == 0
    )

    return {
        "passed": passed,
        "char_count": char_count,
        "word_count": word_count,
        "section_count": (
            section_count
        ),
        "alpha_ratio": round(
            alpha_ratio,
            4,
        ),
        "reasons": reasons,
    }


# ============================================================
# Load Parsed Document
# ============================================================

def _load_json(
    path: Path,
) -> dict:
    """
    JSON 로드.
    """

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# Clean One Document
# ============================================================

def clean_document(
    parsed_path: Path,
) -> dict:
    """
    parsed_document.json 한 편 정제.

    parser output
        ↓
    clean title
    clean abstract
    clean sections
        ↓
    clean_content
        ↓
    quality
        ↓
    SHA-256
    """

    parsed = _load_json(
        parsed_path
    )

    title = _remove_inline_residue(
        parsed.get(
            "title",
            "",
        )
    )

    abstract = _remove_inline_residue(
        parsed.get(
            "abstract",
            "",
        )
    )

    original_sections = parsed.get(
        "sections",
        [],
    )

    cleaned_sections, cleaning_stats = (
        _clean_sections(
            original_sections
        )
    )

    clean_content = _render_clean_content(
        title=title,
        abstract=abstract,
        sections=cleaned_sections,
    )

    quality = _quality_check(
        clean_content,
        cleaned_sections,
    )

    content_hash = _content_hash(
        clean_content
    )

    raw_stats = parsed.get(
        "stats",
        {},
    )

    before_chars = raw_stats.get(
        "char_count",
        0,
    )

    before_words = raw_stats.get(
        "word_count",
        0,
    )

    after_chars = quality[
        "char_count"
    ]

    after_words = quality[
        "word_count"
    ]

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
        "title": title,
        "abstract": abstract,

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

        "quality": quality,

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
    """
    각 논문 폴더:

    clean_content.txt
    cleaned_document.json

    생성.
    """

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
        "title": cleaned[
            "title"
        ],

        "abstract": cleaned[
            "abstract"
        ],

        "sections": cleaned[
            "sections"
        ],

        "content_hash": cleaned[
            "content_hash"
        ],

        "normalization_version": (
            cleaned[
                "normalization_version"
            ]
        ),

        "quality": cleaned[
            "quality"
        ],

        "cleaning_stats": cleaned[
            "cleaning_stats"
        ],
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Find Parsed Documents
# ============================================================

def _find_parsed_documents() -> list[Path]:
    """
    parser가 생성한 parsed_document.json 모두 탐색.
    """

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            f"Resolved root not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/parsed_document.json"
        )
    )


# ============================================================
# Global Duplicate Check
# ============================================================

def _check_duplicate_hashes(
    results: list[dict],
) -> None:
    """
    같은 clean_content SHA-256을 가진 논문 확인.

    결과 record에 duplicate_of 추가.
    """

    seen: dict[str, str] = {}

    for result in results:

        if (
            result.get(
                "status"
            )
            != "success"
        ):
            continue

        content_hash = result.get(
            "content_hash"
        )

        source_id = result.get(
            "source_id"
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
# Save Global Report
# ============================================================

def _save_report(
    results: list[dict],
) -> None:
    """
    전체 cleaning report.
    """

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
        if result.get(
            "status"
        )
        == "success"
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
        if result.get(
            "status"
        )
        == "success"
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
        if result.get(
            "status"
        )
        == "success"
    )

    payload = {
        "total": len(
            results
        ),

        "success": success_count,
        "failed": failed_count,

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

        "documents": results,
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
    """
    arXiv 파일럿 15편:

    parsed_document.json
            ↓
    references 제거
            ↓
    noise normalization
            ↓
    section 구조 보존
            ↓
    clean_content.txt
            ↓
    quality check
            ↓
    SHA-256
            ↓
    cleaned_document.json

    아직 AWS INSERT는 하지 않는다.
    """

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv Cleaner"
    )

    print("=" * 70)

    print(
        f"Input root : "
        f"{RESOLVED_ROOT}"
    )

    parsed_files = (
        _find_parsed_documents()
    )

    print(
        f"Documents  : "
        f"{len(parsed_files)}"
    )

    print(
        f"Version    : "
        f"{settings.normalization_version}"
    )

    if not parsed_files:

        print(
            "No parsed_document.json files found."
        )

        return

    results: list[dict] = []

    # ========================================================
    # Clean each paper
    # ========================================================

    for index, parsed_path in enumerate(
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
            f"[{index}/{len(parsed_files)}]"
        )

        print(
            f"[AXIS] {topic_axis}"
        )

        print(
            f"[ID]   {source_id}"
        )

        print("-" * 70)

        try:

            cleaned = clean_document(
                parsed_path
            )

            _save_cleaned_document(
                paper_dir,
                cleaned,
            )

            quality = cleaned[
                "quality"
            ]

            stats = cleaned[
                "cleaning_stats"
            ]

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
                    f"[QUALITY] Reasons: "
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
                    "status": "success",

                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "title": cleaned[
                        "title"
                    ],

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

                    "quality": quality,

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
                    "status": "failed",
                    "topic_axis": (
                        topic_axis
                    ),
                    "source_id": (
                        source_id
                    ),
                    "error": str(
                        exc
                    ),
                }
            )

        # 한 편 처리할 때마다 report 저장
        _save_report(
            results
        )

    # ========================================================
    # Final Duplicate Check
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
        if item.get(
            "status"
        )
        == "success"
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
        if item.get(
            "status"
        )
        == "success"
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
        if item.get(
            "status"
        )
        == "success"
    )

    print()
    print("=" * 70)

    print(
        "Cleaning completed."
    )

    print("=" * 70)

    print(
        f"Total        : "
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


if __name__ == "__main__":
    main()