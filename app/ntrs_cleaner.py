import hashlib
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
    / "ntrs_cleaning_report.json"
)


# ============================================================
# Normalization Version
# ============================================================
#
# arXiv와 동일하게 v1.
#
# 나중에 cleaning rule을 변경하면
# v2로 올린다.
# ============================================================

NORMALIZATION_VERSION = "v1"


# ============================================================
# Quality Thresholds
# ============================================================
#
# arXiv Core와 같은 기준을 사용한다.
#
# ============================================================

MIN_CHAR_COUNT = 5_000
MIN_WORD_COUNT = 800
MIN_ALPHA_RATIO = 0.45
MIN_SECTION_COUNT = 2


# ============================================================
# Remove Sections
# ============================================================
#
# 논문 본문 자체가 아니라 publication residue에 가까운
# section들.
#
# References가 시작되면 뒤쪽 bibliography 전체를
# clean body에서는 제외한다.
#
# ============================================================

REFERENCE_HEADINGS = {
    "reference",
    "references",
    "bibliography",
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
# Text Normalization
# ============================================================

def _normalize_text(
    text: str,
) -> str:
    """
    의미를 바꾸지 않는 범위에서만 normalize.

    하지 않는 것:
    - lowercase
    - stemming
    - stopword 제거
    - 숫자 제거
    - 문장 재작성

    Transformer와 RAG에 원래 학술문장을 남겨야 한다.
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

    # line 내부 중복 space
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text,
    )

    # punctuation 앞의 이상한 space 일부 정리
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    # 괄호 바로 안쪽의 이상한 공백
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

    # 과도한 빈 줄
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
    """
    예:

        VII. REFERENCES
        7 References
        7. REFERENCES

    모두:

        references

    로 비교 가능하게 만든다.
    """

    heading = (
        _normalize_text(
            heading
        )
        .strip()
    )

    # --------------------------------------------------------
    # Leading section numbers
    #
    # 1.
    # 1.2
    # VII.
    # A.
    # --------------------------------------------------------

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

    return (
        normalized
        in REFERENCE_HEADINGS
    )


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

    # page 1 / page 1 of 10
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
    확실한 residue만 제거.

    애매한 것은 보존한다.
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
    # Caption은 지나치게 짧으면 의미가 없다.
    # 기존 arXiv cleaner와 같은 원칙.
    # --------------------------------------------------------

    if (
        block_type
        == "figure_caption"
        and len(
            stripped
        ) < 20
    ):

        return True

    # --------------------------------------------------------
    # paragraph가 지나치게 짧고 알파벳도 거의 없으면
    # TXT conversion residue 가능성이 높다.
    #
    # 단, 짧다는 이유만으로 일반 영어 문장을 지우지는 않는다.
    # --------------------------------------------------------

    if (
        block_type
        == "paragraph"
        and len(
            stripped
        ) < 8
    ):

        alpha_count = sum(
            1
            for char in stripped
            if char.isalpha()
        )

        if alpha_count < 3:

            return True

    return False


# ============================================================
# Caption Label Cleanup
# ============================================================

def _clean_caption(
    text: str,
) -> str:
    """
    Fig. 1: ...
    Figure 2. ...
    Table IV: ...

    같은 label만 제거.

    caption 내용은 유지한다.
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

    if _is_probable_junk(
        text,
        block_type,
    ):

        return None

    if (
        block_type
        == "figure_caption"
    ):

        text = (
            _clean_caption(
                text
            )
        )

        if len(
            text
        ) < 20:

            return None

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
# Exact Consecutive Duplicate Removal
# ============================================================

def _remove_consecutive_duplicates(
    blocks: list[dict],
) -> list[dict]:
    """
    매우 보수적인 dedup.

    같은 block이 바로 연속해서 반복된 경우에만 제거한다.

    NASA converted TXT에서 실제로 같은 문장이
    다른 section에서 의미 있게 재등장할 수도 있으므로
    global dedup은 하지 않는다.
    """

    result: list[
        dict
    ] = []

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

        previous_key = (
            key
        )

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
    """
    핵심 cleaner.

    References가 시작되면 이후는 clean body에서 제외.

    Acknowledgments/Funding/Conflict 등은
    해당 section만 제거한다.
    """

    clean_sections: list[
        dict
    ] = []

    removed_sections = []

    references_removed = False

    original_section_count = len(
        sections
    )

    for section in sections:

        heading = (
            _normalize_text(
                section.get(
                    "heading",
                    "Document Body",
                )
            )
        )

        # ====================================================
        # References boundary
        # ====================================================

        if _is_reference_heading(
            heading
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

            # bibliography 뒤는 V1 clean body에서 제외
            break

        # ====================================================
        # Non-body sections
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
                []
            )
        )

        if not isinstance(
            blocks,
            list,
        ):

            continue

        clean_blocks: list[
            dict
        ] = []

        for block in blocks:

            if not isinstance(
                block,
                dict,
            ):

                continue

            cleaned = (
                _clean_block(
                    block
                )
            )

            if (
                cleaned
                is not None
            ):

                clean_blocks.append(
                    cleaned
                )

        clean_blocks = (
            _remove_consecutive_duplicates(
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

        "clean_section_count": len(
            clean_sections
        ),

        "references_removed": (
            references_removed
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
# Renderer
# ============================================================

def _render_clean_content(
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:

    output: list[
        str
    ] = []

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

            block_type = (
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
# SHA-256
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
# Quality
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
    ) = _clean_sections(
        sections
    )

    # ========================================================
    # Render
    # ========================================================

    clean_content = (
        _render_clean_content(
            title=title,
            abstract=abstract,
            sections=clean_sections,
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

            "section_count": len(
                sections
            ),
        },

        "after": {
            "char_count": (
                after_chars
            ),

            "word_count": (
                after_words
            ),

            "section_count": len(
                clean_sections
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

    result = {
        "source": "ntrs",

        "source_id": (
            source_info.get(
                "source_id"
            )
            or paper_dir.name
        ),

        "topic_axis": (
            source_info.get(
                "topic_axis"
            )
            or paper_dir.parent.name
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
# Save Document
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

    json_payload = dict(
        cleaned
    )

    # clean_content는 txt에도 있으므로
    # JSON에도 남겨 loader가 단독으로 읽을 수 있게 한다.
    _save_json(
        json_path,
        json_payload,
    )


# ============================================================
# Local Content-hash Dedup
# ============================================================

def _mark_local_duplicates(
    results: list[dict],
) -> int:
    """
    NTRS 15편 내부 hash 중복 검사.

    DB 적재 단계에서는 기존 arXiv 15편의 hash와도
    다시 비교한다.

    여기서는 NTRS 내부 duplicate만 검사.
    """

    seen: dict[
        str,
        dict
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

            result[
                "duplicate"
            ] = False

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
    results: list[dict],
    duplicate_count: int,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = sum(
        1
        for item in results
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
        for item in results
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
        item.get(
            "words_before",
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

    words_after = sum(
        item.get(
            "words_after",
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

    chars_before = sum(
        item.get(
            "chars_before",
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

    chars_after = sum(
        item.get(
            "chars_after",
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
        "source": "ntrs",

        "normalization_version": (
            NORMALIZATION_VERSION
        ),

        "total": len(
            results
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

        "duplicate_count": (
            duplicate_count
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

        "documents": (
            results
        ),
    }

    _save_json(
        REPORT_FILE,
        payload,
    )


# ============================================================
# Find Parsed Documents
# ============================================================

def _find_documents() -> list[
    Path
]:

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            f"NTRS resolved root "
            f"not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/parsed_document.json"
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - NASA NTRS Cleaner"
    )

    print("=" * 70)

    documents = (
        _find_documents()
    )

    print(
        f"Input root : "
        f"{RESOLVED_ROOT}"
    )

    print(
        f"Documents  : "
        f"{len(documents)}"
    )

    print(
        f"Version    : "
        f"{NORMALIZATION_VERSION}"
    )

    if not documents:

        print()
        print(
            "[STOP] No parsed_document.json "
            "files found."
        )

        return

    results = []

    # ========================================================
    # Clean
    # ========================================================

    for index, parsed_path in enumerate(
        documents,
        start=1,
    ):

        paper_dir = (
            parsed_path.parent
        )

        source_id = (
            paper_dir.name
        )

        topic_axis = (
            paper_dir.parent.name
        )

        print()
        print(
            "-" * 70
        )

        print(
            f"[{index}/"
            f"{len(documents)}]"
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
            "-" * 70
        )

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
                    "statistics"
                ]
            )

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

            results.append(
                {
                    "status": (
                        "success"
                    ),

                    "source_id": (
                        cleaned[
                            "source_id"
                        ]
                    ),

                    "topic_axis": (
                        cleaned[
                            "topic_axis"
                        ]
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

                    "duplicate": False,
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
    # Local Dedup
    # ========================================================

    duplicate_count = (
        _mark_local_duplicates(
            results
        )
    )

    # ========================================================
    # Report
    # ========================================================

    _save_report(
        results,
        duplicate_count,
    )

    success = sum(
        1
        for item in results
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
        for item in results
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
        item.get(
            "words_before",
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

    words_after = sum(
        item.get(
            "words_after",
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

    reduction = (
        _percentage_reduction(
            words_before,
            words_after,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "NTRS CLEANING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Total          : "
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
        f"{reduction}%"
    )

    print(
        f"Report         : "
        f"{REPORT_FILE}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()