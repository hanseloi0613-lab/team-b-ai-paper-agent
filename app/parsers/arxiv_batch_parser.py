import json
import re
import subprocess
import sys
from pathlib import Path

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - arXiv Core-100 Batch Parser
# ============================================================
#
# 현재 대상:
#
# Core Pilot 30편은 이미 완료.
#
# 이번 단계에서는 Human QA를 통과한
# 신규 arXiv 35편만 최종 검증한다.
#
#   rover_autonomy       12
#   onboard_ai           11
#   satellite_autonomy   12
#   -----------------------
#   TOTAL                35
#
#
# Resolver 결과:
#
#   HTML 28
#   TeX   1
#   PDF   6
#
#
# 처리 흐름:
#
# selection manifest
#       ↓
# resolution.json
#       ↓
# selected_format 확인
#       ↓
# html_parser.py
# tex_parser.py
# pdf_parser.py
#       ↓
# raw_content.txt
# parsed_document.json
#       ↓
# 신규 35편 최종 QA
#
#
# 중요:
#
# 기존 parser들은 arxiv_resolved 아래의
# 해당 형식 파일 전체를 탐색할 수 있다.
#
# 따라서 기존 Pilot 15편이 다시 파싱될 수도 있지만
# 최종 PASS/FAIL 판정은 selection manifest의
# 신규 35편만 대상으로 한다.
# ============================================================


PARSER_VERSION = "core100_v1"


# ============================================================
# Paths
# ============================================================

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "selections"
    / "arxiv_core100_selected.json"
)

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
    / "arxiv_core100_parsing_report.json"
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
# Existing Parser Modules
# ============================================================

PARSER_MODULES = {
    "html": "app.parsers.html_parser",
    "tex": "app.parsers.tex_parser",
    "pdf": "app.parsers.pdf_parser",
}


# ============================================================
# JSON Helpers
# ============================================================

def _load_json(
    path: Path,
) -> dict:
    """
    JSON 파일 로드.
    """

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
    """
    JSON 파일 저장.
    """

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
# arXiv ID Helper
# ============================================================

def _base_arxiv_id(
    source_id: str,
) -> str:
    """
    버전 제거.

    예:

        2410.09126v1
            ↓
        2410.09126
    """

    return re.sub(
        r"v\d+$",
        "",
        source_id.strip(),
    )


# ============================================================
# Selection Manifest
# ============================================================

def _load_selection() -> dict[
    str,
    list[str],
]:
    """
    Human QA 결과로 확정된 신규 35편을 읽는다.

    data/selections/
        arxiv_core100_selected.json
    """

    payload = _load_json(
        SELECTION_FILE
    )

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
            "Selection JSON does not contain "
            "'selected_documents'."
        )

    validated: dict[
        str,
        list[str],
    ] = {}

    seen_base_ids: set[str] = set()

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

        clean_ids: list[str] = []

        for source_id in source_ids:

            source_id = str(
                source_id
            ).strip()

            if not source_id:

                raise RuntimeError(
                    f"{topic_axis}: "
                    "empty source_id found."
                )

            base_id = (
                _base_arxiv_id(
                    source_id
                )
            )

            if base_id in seen_base_ids:

                raise RuntimeError(
                    "Duplicate arXiv paper "
                    f"in selection: {source_id}"
                )

            seen_base_ids.add(
                base_id
            )

            clean_ids.append(
                source_id
            )

        if (
            len(clean_ids)
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count}, "
                f"found {len(clean_ids)}"
            )

        validated[
            topic_axis
        ] = clean_ids

    actual_total = sum(
        len(source_ids)
        for source_ids
        in validated.values()
    )

    if actual_total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected total "
            f"{EXPECTED_TOTAL}, "
            f"found {actual_total}."
        )

    declared_total = (
        payload.get(
            "selected_count"
        )
    )

    if (
        declared_total is not None
        and int(
            declared_total
        )
        != actual_total
    ):

        raise RuntimeError(
            "selected_count mismatch: "
            f"declared={declared_total}, "
            f"actual={actual_total}"
        )

    return validated


# ============================================================
# Find Resolved Paper Directory
# ============================================================

def _find_paper_dir(
    topic_axis: str,
    source_id: str,
) -> Path:
    """
    기본적으로:

        data/tmp/arxiv_resolved/
            <axis>/<source_id>/

    를 찾는다.

    혹시 resolver가 같은 논문의 다른 version ID를
    directory 이름으로 사용한 경우를 대비하여
    base arXiv ID fallback도 수행한다.
    """

    axis_dir = (
        RESOLVED_ROOT
        / topic_axis
    )

    exact_dir = (
        axis_dir
        / source_id
    )

    if exact_dir.exists():

        return exact_dir

    if not axis_dir.exists():

        raise FileNotFoundError(
            f"Axis directory not found: "
            f"{axis_dir}"
        )

    wanted_base = (
        _base_arxiv_id(
            source_id
        )
    )

    candidates = []

    for child in axis_dir.iterdir():

        if not child.is_dir():

            continue

        child_base = (
            _base_arxiv_id(
                child.name
            )
        )

        if child_base == wanted_base:

            candidates.append(
                child
            )

    if len(candidates) == 1:

        return candidates[0]

    if len(candidates) > 1:

        raise RuntimeError(
            "Multiple resolved directories "
            f"match {source_id}: "
            f"{[item.name for item in candidates]}"
        )

    raise FileNotFoundError(
        "Resolved paper directory "
        f"not found: "
        f"{topic_axis} / {source_id}"
    )


# ============================================================
# Load Resolved Documents
# ============================================================

def _load_resolved_documents(
    selection: dict[
        str,
        list[str],
    ],
) -> list[dict]:
    """
    선택된 신규 35편에 대해:

    - paper directory 존재
    - resolution.json 존재
    - resolver success
    - selected_format 유효
    - 실제 source file 존재

    를 검사한다.
    """

    documents: list[dict] = []

    errors: list[dict] = []

    for (
        topic_axis,
        source_ids,
    ) in selection.items():

        for source_id in source_ids:

            try:

                paper_dir = (
                    _find_paper_dir(
                        topic_axis,
                        source_id,
                    )
                )

            except Exception as exc:

                errors.append(
                    {
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

                continue

            resolution_path = (
                paper_dir
                / "resolution.json"
            )

            metadata_path = (
                paper_dir
                / "metadata.json"
            )

            if not resolution_path.exists():

                errors.append(
                    {
                        "topic_axis": (
                            topic_axis
                        ),

                        "source_id": (
                            source_id
                        ),

                        "error": (
                            "resolution.json missing"
                        ),
                    }
                )

                continue

            try:

                resolution = (
                    _load_json(
                        resolution_path
                    )
                )

            except Exception as exc:

                errors.append(
                    {
                        "topic_axis": (
                            topic_axis
                        ),

                        "source_id": (
                            source_id
                        ),

                        "error": (
                            "Invalid resolution.json: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

                continue

            if not resolution.get(
                "success",
                False,
            ):

                errors.append(
                    {
                        "topic_axis": (
                            topic_axis
                        ),

                        "source_id": (
                            source_id
                        ),

                        "error": (
                            "resolution success=false"
                        ),
                    }
                )

                continue

            selected_format = str(
                resolution.get(
                    "selected_format",
                    "",
                )
            ).strip().lower()

            if (
                selected_format
                not in PARSER_MODULES
            ):

                errors.append(
                    {
                        "topic_axis": (
                            topic_axis
                        ),

                        "source_id": (
                            source_id
                        ),

                        "error": (
                            "Unsupported "
                            "selected_format: "
                            f"{selected_format}"
                        ),
                    }
                )

                continue

            # =================================================
            # Source File
            # =================================================

            if selected_format == "html":

                source_file = (
                    paper_dir
                    / "paper.html"
                )

            elif selected_format == "tex":

                source_file = (
                    paper_dir
                    / "paper.tex"
                )

            else:

                source_file = (
                    paper_dir
                    / "paper.pdf"
                )

            if not source_file.exists():

                errors.append(
                    {
                        "topic_axis": (
                            topic_axis
                        ),

                        "source_id": (
                            source_id
                        ),

                        "error": (
                            f"{source_file.name} "
                            "missing"
                        ),
                    }
                )

                continue

            # =================================================
            # Title
            # =================================================

            title = str(
                resolution.get(
                    "title",
                    "",
                )
                or ""
            ).strip()

            if (
                not title
                and metadata_path.exists()
            ):

                try:

                    metadata = (
                        _load_json(
                            metadata_path
                        )
                    )

                    title = str(
                        metadata.get(
                            "title",
                            "",
                        )
                        or ""
                    ).strip()

                except Exception:

                    pass

            documents.append(
                {
                    "topic_axis": (
                        topic_axis
                    ),

                    "requested_source_id": (
                        source_id
                    ),

                    "resolved_source_id": (
                        paper_dir.name
                    ),

                    "title": (
                        title
                    ),

                    "selected_format": (
                        selected_format
                    ),

                    "paper_dir": (
                        paper_dir
                    ),

                    "source_file": (
                        source_file
                    ),
                }
            )

    if errors:

        print()
        print("=" * 70)

        print(
            "RESOLVED DOCUMENT "
            "VALIDATION FAILED"
        )

        print("=" * 70)

        for item in errors:

            print()

            print(
                f"[ERROR] "
                f"{item['topic_axis']} / "
                f"{item['source_id']}"
            )

            print(
                f"        "
                f"{item['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} selected "
            "document(s) failed "
            "resolver validation."
        )

    if (
        len(documents)
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL} "
            f"resolved documents, "
            f"found "
            f"{len(documents)}."
        )

    return documents


# ============================================================
# Format Counts
# ============================================================

def _format_counts(
    documents: list[dict],
) -> dict[str, int]:

    counts = {
        "html": 0,
        "tex": 0,
        "pdf": 0,
    }

    for document in documents:

        selected_format = (
            document[
                "selected_format"
            ]
        )

        counts[
            selected_format
        ] += 1

    return counts


# ============================================================
# Run Existing Parser
# ============================================================

def _run_parser_module(
    selected_format: str,
) -> dict:
    """
    현재 사용 중인 Python interpreter로
    기존 parser module을 실행한다.

    예:

        python -m app.parsers.html_parser
    """

    module_name = (
        PARSER_MODULES[
            selected_format
        ]
    )

    print()
    print("=" * 70)

    print(
        f"RUN "
        f"{selected_format.upper()} "
        f"PARSER"
    )

    print("=" * 70)

    print(
        f"Module : "
        f"{module_name}"
    )

    print(
        f"Python : "
        f"{sys.executable}"
    )

    print(
        f"CWD    : "
        f"{PROJECT_ROOT}"
    )

    print()

    completed = (
        subprocess.run(
            [
                sys.executable,
                "-m",
                module_name,
            ],
            cwd=str(
                PROJECT_ROOT
            ),
            check=False,
        )
    )

    return {
        "format": (
            selected_format
        ),

        "module": (
            module_name
        ),

        "return_code": (
            completed.returncode
        ),

        "success": (
            completed.returncode
            == 0
        ),
    }


# ============================================================
# Parsed Output Validation
# ============================================================

def _validate_parsed_document(
    document: dict,
) -> dict:
    """
    한 논문의 parser 산출물을 검사한다.

    필수:

        raw_content.txt
        parsed_document.json

    parsed_document.json:

        title
        abstract
        sections
        stats
    """

    topic_axis = (
        document[
            "topic_axis"
        ]
    )

    requested_source_id = (
        document[
            "requested_source_id"
        ]
    )

    resolved_source_id = (
        document[
            "resolved_source_id"
        ]
    )

    selected_format = (
        document[
            "selected_format"
        ]
    )

    paper_dir = (
        document[
            "paper_dir"
        ]
    )

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    parsed_path = (
        paper_dir
        / "parsed_document.json"
    )

    result = {
        "topic_axis": (
            topic_axis
        ),

        "source_id": (
            requested_source_id
        ),

        "resolved_source_id": (
            resolved_source_id
        ),

        "selected_format": (
            selected_format
        ),

        "resolver_title": (
            document.get(
                "title",
                "",
            )
        ),

        "status": (
            "failed"
        ),

        "raw_content_exists": (
            raw_path.exists()
        ),

        "parsed_document_exists": (
            parsed_path.exists()
        ),
    }

    # ========================================================
    # Required Files
    # ========================================================

    if not raw_path.exists():

        result[
            "error"
        ] = (
            "raw_content.txt missing"
        )

        return result

    if not parsed_path.exists():

        result[
            "error"
        ] = (
            "parsed_document.json "
            "missing"
        )

        return result

    # ========================================================
    # Raw Content
    # ========================================================

    try:

        raw_content = (
            raw_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            .strip()
        )

    except Exception as exc:

        result[
            "error"
        ] = (
            "Cannot read "
            "raw_content.txt: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return result

    if not raw_content:

        result[
            "error"
        ] = (
            "raw_content.txt is empty"
        )

        return result

    # ========================================================
    # Parsed JSON
    # ========================================================

    try:

        parsed = (
            _load_json(
                parsed_path
            )
        )

    except Exception as exc:

        result[
            "error"
        ] = (
            "Invalid "
            "parsed_document.json: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return result

    title = str(
        parsed.get(
            "title",
            "",
        )
        or ""
    ).strip()

    abstract = str(
        parsed.get(
            "abstract",
            "",
        )
        or ""
    ).strip()

    sections = (
        parsed.get(
            "sections"
        )
    )

    stats = (
        parsed.get(
            "stats"
        )
    )

    # ========================================================
    # Schema Check
    # ========================================================

    if not title:

        result[
            "error"
        ] = (
            "parsed title is empty"
        )

        return result

    if not isinstance(
        sections,
        list,
    ):

        result[
            "error"
        ] = (
            "sections is not a list"
        )

        return result

    if not isinstance(
        stats,
        dict,
    ):

        result[
            "error"
        ] = (
            "stats is not an object"
        )

        return result

    # ========================================================
    # Actual Statistics
    # ========================================================

    actual_char_count = len(
        raw_content
    )

    actual_word_count = len(
        raw_content.split()
    )

    section_count = len(
        sections
    )

    # 너무 짧은 경우 즉시 실패시키지는 않는다.
    # Cleaner Quality Gate에서 더 엄격히 평가한다.
    #
    # 여기서는 parser 자체가 산출물을 만들었는지가 핵심.
    # ========================================================

    result.update(
        {
            "status": (
                "success"
            ),

            "parsed_title": (
                title
            ),

            "abstract_chars": (
                len(
                    abstract
                )
            ),

            "section_count": (
                section_count
            ),

            "word_count": (
                actual_word_count
            ),

            "char_count": (
                actual_char_count
            ),

            "stats": (
                stats
            ),
        }
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv Core-100 "
        "Batch Parser"
    )

    print("=" * 70)

    print(
        f"Version        : "
        f"{PARSER_VERSION}"
    )

    print(
        f"Selection file : "
        f"{SELECTION_FILE}"
    )

    print(
        f"Resolved root  : "
        f"{RESOLVED_ROOT}"
    )

    # ========================================================
    # 1. Precheck
    # ========================================================

    try:

        selection = (
            _load_selection()
        )

        documents = (
            _load_resolved_documents(
                selection
            )
        )

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            "BATCH PARSER "
            "PRECHECK FAILED"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("=" * 70)

        return

    # ========================================================
    # 2. Format Distribution
    # ========================================================

    counts = (
        _format_counts(
            documents
        )
    )

    print()
    print(
        "Selected documents:"
    )

    print(
        f"  HTML : "
        f"{counts['html']}"
    )

    print(
        f"  TeX  : "
        f"{counts['tex']}"
    )

    print(
        f"  PDF  : "
        f"{counts['pdf']}"
    )

    print(
        f"  TOTAL: "
        f"{len(documents)}"
    )

    print()

    for topic_axis in (
        "rover_autonomy",
        "onboard_ai",
        "satellite_autonomy",
    ):

        axis_count = sum(
            1
            for document
            in documents
            if document[
                "topic_axis"
            ] == topic_axis
        )

        print(
            f"  {topic_axis:22} : "
            f"{axis_count}"
        )

    # ========================================================
    # 3. Run Existing Parsers
    # ========================================================

    parser_runs: list[dict] = []

    for selected_format in (
        "html",
        "tex",
        "pdf",
    ):

        if (
            counts[
                selected_format
            ]
            == 0
        ):

            print()
            print(
                f"[SKIP] "
                f"No selected "
                f"{selected_format.upper()} "
                f"documents."
            )

            continue

        parser_result = (
            _run_parser_module(
                selected_format
            )
        )

        parser_runs.append(
            parser_result
        )

        if not parser_result[
            "success"
        ]:

            print()
            print(
                f"[WARNING] "
                f"{selected_format.upper()} "
                f"parser returned "
                f"code "
                f"{parser_result['return_code']}."
            )

            print(
                "Final document QA will "
                "determine actual failures."
            )

    # ========================================================
    # 4. Validate Selected 35 Only
    # ========================================================

    print()
    print("=" * 70)

    print(
        "VALIDATING SELECTED "
        "35 DOCUMENTS"
    )

    print("=" * 70)

    results: list[dict] = []

    for (
        index,
        document,
    ) in enumerate(
        documents,
        start=1,
    ):

        result = (
            _validate_parsed_document(
                document
            )
        )

        results.append(
            result
        )

        print()
        print("-" * 70)

        print(
            f"[{index:02d}/"
            f"{len(documents)}]"
        )

        print(
            f"[AXIS] "
            f"{result['topic_axis']}"
        )

        print(
            f"[ID]   "
            f"{result['source_id']}"
        )

        if (
            result[
                "resolved_source_id"
            ]
            != result[
                "source_id"
            ]
        ):

            print(
                f"[DIR]  "
                f"{result['resolved_source_id']}"
            )

        print(
            f"[FMT]  "
            f"{result['selected_format']}"
        )

        if (
            result[
                "status"
            ]
            == "success"
        ):

            print(
                f"[OK] Title    : "
                f"{result['parsed_title']}"
            )

            print(
                f"[OK] Sections : "
                f"{result['section_count']}"
            )

            print(
                f"[OK] Words    : "
                f"{result['word_count']}"
            )

            print(
                f"[OK] Chars    : "
                f"{result['char_count']}"
            )

        else:

            print(
                f"[FAIL] "
                f"{result.get('error')}"
            )

    # ========================================================
    # 5. Summary
    # ========================================================

    success_count = sum(
        1
        for item
        in results
        if item[
            "status"
        ] == "success"
    )

    failed_count = (
        len(results)
        - success_count
    )

    format_success = {
        "html": 0,
        "tex": 0,
        "pdf": 0,
    }

    format_failed = {
        "html": 0,
        "tex": 0,
        "pdf": 0,
    }

    total_words = 0
    total_chars = 0

    min_words = None
    max_words = None

    min_chars = None
    max_chars = None

    for item in results:

        selected_format = (
            item[
                "selected_format"
            ]
        )

        if (
            item[
                "status"
            ]
            == "success"
        ):

            format_success[
                selected_format
            ] += 1

            word_count = int(
                item.get(
                    "word_count",
                    0,
                )
            )

            char_count = int(
                item.get(
                    "char_count",
                    0,
                )
            )

            total_words += (
                word_count
            )

            total_chars += (
                char_count
            )

            if (
                min_words is None
                or word_count
                < min_words
            ):

                min_words = (
                    word_count
                )

            if (
                max_words is None
                or word_count
                > max_words
            ):

                max_words = (
                    word_count
                )

            if (
                min_chars is None
                or char_count
                < min_chars
            ):

                min_chars = (
                    char_count
                )

            if (
                max_chars is None
                or char_count
                > max_chars
            ):

                max_chars = (
                    char_count
                )

        else:

            format_failed[
                selected_format
            ] += 1

    # ========================================================
    # 6. Per-Axis Summary
    # ========================================================

    axis_summary = {}

    for topic_axis in (
        "rover_autonomy",
        "onboard_ai",
        "satellite_autonomy",
    ):

        axis_items = [
            item
            for item
            in results
            if item[
                "topic_axis"
            ] == topic_axis
        ]

        axis_success = sum(
            1
            for item
            in axis_items
            if item[
                "status"
            ] == "success"
        )

        axis_summary[
            topic_axis
        ] = {
            "total": (
                len(axis_items)
            ),

            "success": (
                axis_success
            ),

            "failed": (
                len(axis_items)
                - axis_success
            ),
        }

    # ========================================================
    # 7. Save Report
    # ========================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "parser_version": (
            PARSER_VERSION
        ),

        "selection_file": (
            str(
                SELECTION_FILE
            )
        ),

        "resolved_root": (
            str(
                RESOLVED_ROOT
            )
        ),

        "selected_total": (
            len(documents)
        ),

        "expected_total": (
            EXPECTED_TOTAL
        ),

        "resolved_formats": (
            counts
        ),

        "axis_summary": (
            axis_summary
        ),

        "parser_runs": (
            parser_runs
        ),

        "success": (
            success_count
        ),

        "failed": (
            failed_count
        ),

        "format_success": (
            format_success
        ),

        "format_failed": (
            format_failed
        ),

        "total_words": (
            total_words
        ),

        "total_chars": (
            total_chars
        ),

        "min_words": (
            min_words
        ),

        "max_words": (
            max_words
        ),

        "min_chars": (
            min_chars
        ),

        "max_chars": (
            max_chars
        ),

        "documents": (
            results
        ),
    }

    _save_json(
        REPORT_FILE,
        report,
    )

    # ========================================================
    # 8. Final Console Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "ARXIV CORE-100 "
        "BATCH PARSING COMPLETED"
    )

    print("=" * 70)

    print(
        f"Selected : "
        f"{len(documents)}"
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
        f"  HTML : "
        f"{counts['html']}"
    )

    print(
        f"  TeX  : "
        f"{counts['tex']}"
    )

    print(
        f"  PDF  : "
        f"{counts['pdf']}"
    )

    print()

    print(
        "Parser success:"
    )

    print(
        f"  HTML : "
        f"{format_success['html']} / "
        f"{counts['html']}"
    )

    print(
        f"  TeX  : "
        f"{format_success['tex']} / "
        f"{counts['tex']}"
    )

    print(
        f"  PDF  : "
        f"{format_success['pdf']} / "
        f"{counts['pdf']}"
    )

    print()

    print(
        "Axis QA:"
    )

    for (
        topic_axis,
        summary,
    ) in axis_summary.items():

        print(
            f"  {topic_axis:22} : "
            f"{summary['success']} / "
            f"{summary['total']}"
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

    if (
        min_words is not None
    ):

        print(
            f"Word range  : "
            f"{min_words} ~ "
            f"{max_words}"
        )

    if (
        min_chars is not None
    ):

        print(
            f"Char range  : "
            f"{min_chars} ~ "
            f"{max_chars}"
        )

    print()

    print(
        f"Report      : "
        f"{REPORT_FILE}"
    )

    print("=" * 70)

    # ========================================================
    # Final Gate
    # ========================================================

    if (
        success_count
        == EXPECTED_TOTAL
        and failed_count
        == 0
        and format_success[
            "html"
        ] == counts[
            "html"
        ]
        and format_success[
            "tex"
        ] == counts[
            "tex"
        ]
        and format_success[
            "pdf"
        ] == counts[
            "pdf"
        ]
    ):

        print(
            "[PASS] All 35 selected "
            "arXiv papers parsed."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Run Cleaner for the "
            "35 new arXiv papers."
        )

    else:

        print(
            "[CHECK] Parsing is not "
            "complete."
        )

        print()

        print(
            "Do NOT run Cleaner yet."
        )

        print(
            "Check failed documents "
            "and parser output above."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()