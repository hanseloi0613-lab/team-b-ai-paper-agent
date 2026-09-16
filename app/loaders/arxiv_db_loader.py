import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - arXiv Core-100 DB Loader
# ============================================================
#
# 현재 단계:
#
# 기존 AWS RDS:
#
#   arXiv       15
#   NTRS        15
#   ----------------
#   TOTAL       30
#
#
# 이번 Loader 대상:
#
# Human QA + Resolver + Parser + Cleaner를 모두 통과한
# 신규 arXiv 35편만 적재한다.
#
#   rover_autonomy       +12
#   onboard_ai           +11
#   satellite_autonomy   +12
#   ------------------------
#   TOTAL                +35
#
#
# 정상적인 최초 실행 결과:
#
#   arXiv       50
#   NTRS        15
#   ----------------
#   TOTAL       65
#
#
# 중요:
#
# - 기존 Pilot 15편은 다시 로드하지 않는다.
# - data/selections/arxiv_core100_selected.json 기준
# - quality.passed == True 필수
# - content_hash 필수
# - DB에 같은 content_hash가 다른 문서로 존재하면 전체 중단
# - 모든 35편이 검증된 뒤에만 COMMIT
# - 다시 실행해도 source+source_id UPSERT로 중복 row 생성 안 됨
# ============================================================


LOADER_VERSION = "core100_v1"


# ============================================================
# Database
# ============================================================

TABLE_SCHEMA = "public"
TABLE_NAME = "core_documents"


# ============================================================
# Expected Selection
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
# Expected First-Run DB State
# ============================================================

EXPECTED_FIRST_RUN_ROWS_BEFORE = 30
EXPECTED_FIRST_RUN_ROWS_AFTER = 65

EXPECTED_FIRST_RUN_ARXIV_BEFORE = 15
EXPECTED_FIRST_RUN_ARXIV_AFTER = 50

EXPECTED_FIRST_RUN_NTRS_BEFORE = 15
EXPECTED_FIRST_RUN_NTRS_AFTER = 15


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

ARXIV_CORE100_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "arxiv"
    / "core100_v1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "arxiv_core100_db_loading_report.json"
)


# ============================================================
# Required DB Columns
# ============================================================

REQUIRED_COLUMNS = {
    "source",
    "source_id",
    "title",
    "abstract",
    "authors",
    "categories",
    "published_at",
    "document_type",
    "doi",
    "url",
    "pdf_url",
    "topic_axis",
    "language",
    "raw_content",
    "clean_content",
    "content_hash",
    "normalization_version",
    "char_count",
    "parse_status",
    "metadata",
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
# arXiv ID
# ============================================================

def _strip_version(
    source_id: str,
) -> str:

    return re.sub(
        r"v\d+$",
        "",
        str(
            source_id
        ).strip(),
    )


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

    result = {}

    seen_base_ids = set()

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

            base_id = (
                _strip_version(
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

        result[
            topic_axis
        ] = normalized_ids

    total = sum(
        len(items)
        for items
        in result.values()
    )

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL} selected papers, "
            f"found {total}."
        )

    return result


# ============================================================
# Paper Directory
# ============================================================

def _find_paper_dir(
    topic_axis: str,
    source_id: str,
) -> Path:

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
        _strip_version(
            source_id
        )
    )

    matches = []

    for child in axis_dir.iterdir():

        if not child.is_dir():
            continue

        if (
            _strip_version(
                child.name
            )
            == wanted_base
        ):

            matches.append(
                child
            )

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple directories match "
            f"{source_id}: "
            f"{[item.name for item in matches]}"
        )

    raise FileNotFoundError(
        "Resolved paper directory "
        f"not found: "
        f"{topic_axis} / {source_id}"
    )


# ============================================================
# Find Selected Cleaned Documents
# ============================================================

def _find_selected_cleaned_documents(
    selection: dict[
        str,
        list[str],
    ],
) -> list[Path]:

    cleaned_files = []

    errors = []

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
                    (
                        topic_axis,
                        source_id,
                        str(exc),
                    )
                )

                continue

            cleaned_path = (
                paper_dir
                / "cleaned_document.json"
            )

            if not cleaned_path.exists():

                errors.append(
                    (
                        topic_axis,
                        source_id,
                        "cleaned_document.json missing",
                    )
                )

                continue

            cleaned_files.append(
                cleaned_path
            )

    if errors:

        print()
        print(
            "Missing / invalid cleaned documents:"
        )

        for (
            topic_axis,
            source_id,
            error,
        ) in errors:

            print(
                f"  {topic_axis} / "
                f"{source_id}"
            )

            print(
                f"    -> {error}"
            )

        raise RuntimeError(
            f"{len(errors)} selected "
            "document(s) failed local precheck."
        )

    if (
        len(cleaned_files)
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL} cleaned documents, "
            f"found {len(cleaned_files)}."
        )

    return cleaned_files


# ============================================================
# Core-100 Metadata Cache
# ============================================================

def _load_axis_metadata(
    topic_axis: str,
) -> list[dict]:

    path = (
        ARXIV_CORE100_CACHE_ROOT
        / topic_axis
        / "_merged_candidates.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Core-100 metadata cache "
            f"not found: {path}"
        )

    payload = (
        _load_json(
            path
        )
    )

    documents = (
        payload.get(
            "documents",
            []
        )
    )

    if not isinstance(
        documents,
        list,
    ):

        raise RuntimeError(
            f"Invalid metadata cache: {path}"
        )

    return documents


def _find_metadata(
    topic_axis: str,
    source_id: str,
) -> dict:

    documents = (
        _load_axis_metadata(
            topic_axis
        )
    )

    wanted_base = (
        _strip_version(
            source_id
        )
    )

    # exact version 우선
    for document in documents:

        candidate_id = str(
            document.get(
                "source_id",
                "",
            )
            or ""
        ).strip()

        if candidate_id == source_id:

            return document

    # base ID fallback
    for document in documents:

        candidate_id = str(
            document.get(
                "source_id",
                "",
            )
            or ""
        ).strip()

        if (
            _strip_version(
                candidate_id
            )
            == wanted_base
        ):

            return document

    raise LookupError(
        "Core-100 metadata not found: "
        f"{topic_axis} / {source_id}"
    )


# ============================================================
# Resolution
# ============================================================

def _load_resolution(
    paper_dir: Path,
) -> dict:

    path = (
        paper_dir
        / "resolution.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"resolution.json missing: "
            f"{path}"
        )

    return _load_json(
        path
    )


# ============================================================
# Parser Provenance
# ============================================================

def _parser_name(
    selected_format: str | None,
) -> str:

    selected_format = str(
        selected_format
        or ""
    ).strip().lower()

    mapping = {
        "html": (
            "arxiv_html_parser_v2"
        ),

        "tex": (
            "arxiv_tex_parser_v1"
        ),

        "pdf": (
            "arxiv_pdf_parser_v1"
        ),
    }

    return mapping.get(
        selected_format,
        "arxiv_parser_unknown",
    )


# ============================================================
# PostgreSQL Schema Inspection
# ============================================================

def _get_column_info(
    conn: psycopg.Connection,
) -> dict[str, dict]:

    sql = """
    SELECT
        column_name,
        data_type,
        udt_name,
        is_nullable,
        column_default
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                TABLE_SCHEMA,
                TABLE_NAME,
            ),
        )

        rows = (
            cur.fetchall()
        )

    if not rows:

        raise RuntimeError(
            f"Table not found: "
            f"{TABLE_SCHEMA}.{TABLE_NAME}\n"
            "Run sql/002_schema.sql first."
        )

    result = {}

    for row in rows:

        (
            column_name,
            data_type,
            udt_name,
            is_nullable,
            column_default,
        ) = row

        result[
            column_name
        ] = {
            "data_type": (
                data_type
            ),

            "udt_name": (
                udt_name
            ),

            "is_nullable": (
                is_nullable
            ),

            "column_default": (
                column_default
            ),
        }

    return result


def _validate_schema(
    column_info: dict[str, dict],
) -> None:

    existing_columns = set(
        column_info.keys()
    )

    missing = (
        REQUIRED_COLUMNS
        - existing_columns
    )

    if missing:

        raise RuntimeError(
            "core_documents schema mismatch.\n"
            f"Missing columns: "
            f"{', '.join(sorted(missing))}"
        )


# ============================================================
# PostgreSQL Type Adaptation
# ============================================================

def _adapt_collection(
    value: list | dict,
    column: dict,
) -> Any:

    data_type = str(
        column.get(
            "data_type",
            "",
        )
        or ""
    ).lower()

    udt_name = str(
        column.get(
            "udt_name",
            "",
        )
        or ""
    ).lower()

    if data_type in {
        "json",
        "jsonb",
    }:

        return Jsonb(
            value
        )

    if (
        data_type == "array"
        or udt_name.startswith(
            "_"
        )
    ):

        if isinstance(
            value,
            list,
        ):

            return value

        return [
            str(
                value
            )
        ]

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# Date
# ============================================================

def _parse_date(
    value: str | None,
) -> date | None:

    if not value:
        return None

    try:

        return date.fromisoformat(
            str(
                value
            )[:10]
        )

    except ValueError:

        return None


# ============================================================
# Build One DB Document
# ============================================================

def _build_document(
    cleaned_path: Path,
) -> dict:

    paper_dir = (
        cleaned_path.parent
    )

    topic_axis = (
        paper_dir
        .parent
        .name
    )

    cleaned = (
        _load_json(
            cleaned_path
        )
    )

    resolution = (
        _load_resolution(
            paper_dir
        )
    )

    source_id = str(
        resolution.get(
            "source_id"
        )
        or paper_dir.name
    ).strip()

    metadata = (
        _find_metadata(
            topic_axis,
            source_id,
        )
    )

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    clean_path = (
        paper_dir
        / "clean_content.txt"
    )

    if not raw_path.exists():

        raise FileNotFoundError(
            f"raw_content.txt missing: "
            f"{raw_path}"
        )

    if not clean_path.exists():

        raise FileNotFoundError(
            f"clean_content.txt missing: "
            f"{clean_path}"
        )

    raw_content = (
        raw_path.read_text(
            encoding="utf-8",
        )
    )

    clean_content = (
        clean_path.read_text(
            encoding="utf-8",
        )
    )

    quality = (
        cleaned.get(
            "quality",
            {},
        )
    )

    if not quality.get(
        "passed",
        False,
    ):

        raise ValueError(
            "Quality check failed: "
            f"{source_id}"
        )

    content_hash = (
        cleaned.get(
            "content_hash"
        )
    )

    if not content_hash:

        raise ValueError(
            "content_hash missing: "
            f"{source_id}"
        )

    if not clean_content.strip():

        raise ValueError(
            "clean_content empty: "
            f"{source_id}"
        )

    if not raw_content.strip():

        raise ValueError(
            "raw_content empty: "
            f"{source_id}"
        )

    # ========================================================
    # Metadata / Provenance
    # ========================================================

    original_metadata = (
        metadata.get(
            "metadata",
            {},
        )
    )

    if not isinstance(
        original_metadata,
        dict,
    ):

        original_metadata = {
            "original_metadata": (
                original_metadata
            )
        }

    selected_format = (
        resolution.get(
            "selected_format"
        )
    )

    db_metadata = {
        **original_metadata,

        "matched_queries": (
            metadata.get(
                "matched_queries",
                [],
            )
        ),

        "source_base_id": (
            metadata.get(
                "source_base_id"
            )
            or _strip_version(
                source_id
            )
        ),

        "content_resolution": {
            "selected_format": (
                selected_format
            ),

            "source_url": (
                resolution.get(
                    "source_url"
                )
            ),

            "repair": (
                resolution.get(
                    "repair"
                )
            ),
        },

        "quality": (
            quality
        ),

        "cleaning_stats": (
            cleaned.get(
                "cleaning_stats",
                {},
            )
        ),

        "pipeline": {
            "selection": (
                "arxiv_core100_selected"
            ),

            "parser": (
                _parser_name(
                    selected_format
                )
            ),

            "normalization_version": (
                cleaned.get(
                    "normalization_version"
                )
            ),

            "loader": (
                LOADER_VERSION
            ),
        },
    }

    return {
        "source": (
            metadata.get(
                "source"
            )
            or "arxiv"
        ),

        "source_id": (
            source_id
        ),

        "title": (
            cleaned.get(
                "title"
            )
            or metadata.get(
                "title"
            )
            or ""
        ),

        "abstract": (
            cleaned.get(
                "abstract"
            )
            or metadata.get(
                "abstract"
            )
            or ""
        ),

        "authors": (
            metadata.get(
                "authors",
                [],
            )
        ),

        "categories": (
            metadata.get(
                "categories",
                [],
            )
        ),

        "published_at": (
            _parse_date(
                metadata.get(
                    "published_at"
                )
            )
        ),

        "document_type": (
            metadata.get(
                "document_type"
            )
            or "arxiv_preprint"
        ),

        "doi": (
            metadata.get(
                "doi"
            )
        ),

        "url": (
            metadata.get(
                "url"
            )
        ),

        "pdf_url": (
            metadata.get(
                "pdf_url"
            )
        ),

        "topic_axis": (
            topic_axis
        ),

        "language": (
            metadata.get(
                "language"
            )
            or "en"
        ),

        "raw_content": (
            raw_content
        ),

        "clean_content": (
            clean_content
        ),

        "content_hash": (
            content_hash
        ),

        "normalization_version": (
            cleaned.get(
                "normalization_version"
            )
            or settings.normalization_version
        ),

        "char_count": len(
            clean_content
        ),

        "parse_status": (
            "success"
        ),

        "metadata": (
            db_metadata
        ),
    }


# ============================================================
# Load Selected Local Documents
# ============================================================

def _load_local_documents(
    selection: dict[
        str,
        list[str],
    ],
) -> list[dict]:

    cleaned_files = (
        _find_selected_cleaned_documents(
            selection
        )
    )

    print(
        f"[LOCAL] Selected cleaned documents: "
        f"{len(cleaned_files)}"
    )

    documents = []

    for (
        index,
        cleaned_path,
    ) in enumerate(
        cleaned_files,
        start=1,
    ):

        document = (
            _build_document(
                cleaned_path
            )
        )

        documents.append(
            document
        )

        print(
            f"[LOCAL {index:02d}/"
            f"{len(cleaned_files)}] "
            f"{document['topic_axis']} | "
            f"{document['source_id']} | "
            f"{document['char_count']} chars"
        )

    if (
        len(documents)
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL} local DB records, "
            f"found {len(documents)}."
        )

    return documents


# ============================================================
# Local Hash QA
# ============================================================

def _validate_local_hashes(
    documents: list[dict],
) -> None:

    seen = {}

    for document in documents:

        content_hash = (
            document[
                "content_hash"
            ]
        )

        source_id = (
            document[
                "source_id"
            ]
        )

        if content_hash in seen:

            raise RuntimeError(
                "Local duplicate content hash:\n"
                f"{seen[content_hash]} "
                f"<-> {source_id}"
            )

        seen[
            content_hash
        ] = source_id


# ============================================================
# Existing Row Check
# ============================================================

def _find_existing_source(
    conn: psycopg.Connection,
    source: str,
    source_id: str,
) -> int | None:

    sql = f"""
    SELECT id
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    WHERE source = %s
      AND source_id = %s
    LIMIT 1
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                source,
                source_id,
            ),
        )

        row = (
            cur.fetchone()
        )

    if row is None:
        return None

    return int(
        row[0]
    )


def _find_hash_owner(
    conn: psycopg.Connection,
    content_hash: str,
) -> tuple[
    int,
    str,
    str,
] | None:

    sql = f"""
    SELECT
        id,
        source,
        source_id
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    WHERE content_hash = %s
    LIMIT 1
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                content_hash,
            ),
        )

        row = (
            cur.fetchone()
        )

    if row is None:
        return None

    return (
        int(
            row[0]
        ),
        str(
            row[1]
        ),
        str(
            row[2]
        ),
    )


# ============================================================
# Upsert
# ============================================================

def _upsert_document(
    conn: psycopg.Connection,
    document: dict,
    column_info: dict[str, dict],
) -> int:

    authors = (
        _adapt_collection(
            document[
                "authors"
            ],
            column_info[
                "authors"
            ],
        )
    )

    categories = (
        _adapt_collection(
            document[
                "categories"
            ],
            column_info[
                "categories"
            ],
        )
    )

    metadata = (
        _adapt_collection(
            document[
                "metadata"
            ],
            column_info[
                "metadata"
            ],
        )
    )

    update_timestamp = ""

    if "updated_at" in column_info:

        update_timestamp = """
            updated_at = NOW(),
        """

    sql = f"""
    INSERT INTO {TABLE_SCHEMA}.{TABLE_NAME} (
        source,
        source_id,
        title,
        abstract,
        authors,
        categories,
        published_at,
        document_type,
        doi,
        url,
        pdf_url,
        topic_axis,
        language,
        raw_content,
        clean_content,
        content_hash,
        normalization_version,
        char_count,
        parse_status,
        metadata
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )

    ON CONFLICT (source, source_id)

    DO UPDATE SET
        title = EXCLUDED.title,
        abstract = EXCLUDED.abstract,
        authors = EXCLUDED.authors,
        categories = EXCLUDED.categories,
        published_at = EXCLUDED.published_at,
        document_type = EXCLUDED.document_type,
        doi = EXCLUDED.doi,
        url = EXCLUDED.url,
        pdf_url = EXCLUDED.pdf_url,
        topic_axis = EXCLUDED.topic_axis,
        language = EXCLUDED.language,
        raw_content = EXCLUDED.raw_content,
        clean_content = EXCLUDED.clean_content,
        content_hash = EXCLUDED.content_hash,
        normalization_version = EXCLUDED.normalization_version,
        char_count = EXCLUDED.char_count,
        parse_status = EXCLUDED.parse_status,
        {update_timestamp}
        metadata = EXCLUDED.metadata

    RETURNING id
    """

    values = (
        document[
            "source"
        ],
        document[
            "source_id"
        ],
        document[
            "title"
        ],
        document[
            "abstract"
        ],
        authors,
        categories,
        document[
            "published_at"
        ],
        document[
            "document_type"
        ],
        document[
            "doi"
        ],
        document[
            "url"
        ],
        document[
            "pdf_url"
        ],
        document[
            "topic_axis"
        ],
        document[
            "language"
        ],
        document[
            "raw_content"
        ],
        document[
            "clean_content"
        ],
        document[
            "content_hash"
        ],
        document[
            "normalization_version"
        ],
        document[
            "char_count"
        ],
        document[
            "parse_status"
        ],
        metadata,
    )

    with conn.cursor() as cur:

        cur.execute(
            sql,
            values,
        )

        row = (
            cur.fetchone()
        )

    if row is None:

        raise RuntimeError(
            "UPSERT did not return id: "
            f"{document['source_id']}"
        )

    return int(
        row[0]
    )


# ============================================================
# Counts
# ============================================================

def _count_core_documents(
    conn: psycopg.Connection,
) -> int:

    sql = f"""
    SELECT COUNT(*)
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        row = (
            cur.fetchone()
        )

    return int(
        row[0]
    )


def _count_source_documents(
    conn: psycopg.Connection,
    source: str,
) -> int:

    sql = f"""
    SELECT COUNT(*)
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    WHERE source = %s
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                source,
            ),
        )

        row = (
            cur.fetchone()
        )

    return int(
        row[0]
    )


# ============================================================
# Verify Selected 35 In DB
# ============================================================

def _verify_selected_documents(
    conn: psycopg.Connection,
    documents: list[dict],
) -> dict:

    missing = []

    hash_mismatch = []

    axis_mismatch = []

    for document in documents:

        sql = f"""
        SELECT
            id,
            content_hash,
            topic_axis
        FROM {TABLE_SCHEMA}.{TABLE_NAME}
        WHERE source = %s
          AND source_id = %s
        LIMIT 1
        """

        with conn.cursor() as cur:

            cur.execute(
                sql,
                (
                    document[
                        "source"
                    ],
                    document[
                        "source_id"
                    ],
                ),
            )

            row = (
                cur.fetchone()
            )

        if row is None:

            missing.append(
                document[
                    "source_id"
                ]
            )

            continue

        (
            _db_id,
            db_hash,
            db_axis,
        ) = row

        if (
            str(
                db_hash
            )
            != document[
                "content_hash"
            ]
        ):

            hash_mismatch.append(
                document[
                    "source_id"
                ]
            )

        if (
            str(
                db_axis
            )
            != document[
                "topic_axis"
            ]
        ):

            axis_mismatch.append(
                document[
                    "source_id"
                ]
            )

    verified_count = (
        len(documents)
        - len(missing)
        - len(hash_mismatch)
        - len(axis_mismatch)
    )

    return {
        "total": (
            len(documents)
        ),

        "verified": (
            verified_count
        ),

        "missing": (
            missing
        ),

        "hash_mismatch": (
            hash_mismatch
        ),

        "axis_mismatch": (
            axis_mismatch
        ),
    }


# ============================================================
# Database Summary
# ============================================================

def _print_database_summary(
    conn: psycopg.Connection,
) -> None:

    sql = f"""
    SELECT
        source,
        topic_axis,
        COUNT(*) AS document_count
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    GROUP BY
        source,
        topic_axis
    ORDER BY
        source,
        topic_axis
    """

    print()
    print("=" * 70)

    print(
        "DATABASE SUMMARY"
    )

    print("=" * 70)

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = (
            cur.fetchall()
        )

    for (
        source,
        topic_axis,
        count,
    ) in rows:

        print(
            f"{source:10} | "
            f"{topic_axis:22} | "
            f"{count}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv Core-100 DB Loader"
    )

    print("=" * 70)

    print(
        f"Version        : "
        f"{LOADER_VERSION}"
    )

    print(
        f"Selection file : "
        f"{SELECTION_FILE}"
    )

    print(
        f"Resolved root  : "
        f"{RESOLVED_ROOT}"
    )

    print(
        f"Metadata root  : "
        f"{ARXIV_CORE100_CACHE_ROOT}"
    )

    # ========================================================
    # Local Preflight
    # ========================================================

    selection = (
        _load_selection()
    )

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
        f"{sum(len(x) for x in selection.values())}"
    )

    documents = (
        _load_local_documents(
            selection
        )
    )

    _validate_local_hashes(
        documents
    )

    print()
    print(
        "[LOCAL] All selected papers "
        "passed loader preflight."
    )

    print(
        f"[LOCAL] Ready for DB: "
        f"{len(documents)}"
    )

    # ========================================================
    # Connect AWS
    # ========================================================

    print()
    print(
        "[DB] Connecting to AWS RDS..."
    )

    results = []

    inserted_count = 0
    updated_count = 0
    duplicate_count = 0
    failed_count = 0

    try:

        with psycopg.connect(
            settings.dsn
        ) as conn:

            print(
                "[DB] Connected."
            )

            # =================================================
            # Schema
            # =================================================

            column_info = (
                _get_column_info(
                    conn
                )
            )

            _validate_schema(
                column_info
            )

            print(
                "[DB] core_documents schema OK."
            )

            # =================================================
            # Before Counts
            # =================================================

            before_count = (
                _count_core_documents(
                    conn
                )
            )

            before_arxiv = (
                _count_source_documents(
                    conn,
                    "arxiv",
                )
            )

            before_ntrs = (
                _count_source_documents(
                    conn,
                    "ntrs",
                )
            )

            print()
            print(
                f"[DB] Rows before  : "
                f"{before_count}"
            )

            print(
                f"[DB] arXiv before : "
                f"{before_arxiv}"
            )

            print(
                f"[DB] NTRS before  : "
                f"{before_ntrs}"
            )

            if (
                before_count
                == EXPECTED_FIRST_RUN_ROWS_BEFORE
            ):

                print(
                    "[DB] First-run state "
                    "detected: 30 rows."
                )

            else:

                print(
                    "[DB] NOTE: DB is not in "
                    "the original 30-row state."
                )

                print(
                    "[DB] UPSERT safety remains active."
                )

            # =================================================
            # UPSERT selected 35 only
            # =================================================

            for (
                index,
                document,
            ) in enumerate(
                documents,
                start=1,
            ):

                source = (
                    document[
                        "source"
                    ]
                )

                source_id = (
                    document[
                        "source_id"
                    ]
                )

                print()
                print("-" * 70)

                print(
                    f"[{index}/"
                    f"{len(documents)}] "
                    f"{document['topic_axis']}"
                )

                print(
                    f"[ID]    "
                    f"{source_id}"
                )

                print(
                    f"[TITLE] "
                    f"{document['title']}"
                )

                try:

                    existing_id = (
                        _find_existing_source(
                            conn,
                            source,
                            source_id,
                        )
                    )

                    hash_owner = (
                        _find_hash_owner(
                            conn,
                            document[
                                "content_hash"
                            ],
                        )
                    )

                    # ==========================================
                    # Same hash owned by ANOTHER document
                    # ==========================================

                    if (
                        hash_owner
                        is not None
                    ):

                        (
                            owner_id,
                            owner_source,
                            owner_source_id,
                        ) = hash_owner

                        same_document = (
                            owner_source
                            == source
                            and
                            owner_source_id
                            == source_id
                        )

                        if not same_document:

                            duplicate_count += 1

                            raise RuntimeError(
                                "Duplicate DB content hash.\n"
                                f"Current: "
                                f"{source}/"
                                f"{source_id}\n"
                                f"Existing DB id: "
                                f"{owner_id}\n"
                                f"Existing: "
                                f"{owner_source}/"
                                f"{owner_source_id}"
                            )

                    # ==========================================
                    # UPSERT
                    # ==========================================

                    db_id = (
                        _upsert_document(
                            conn,
                            document,
                            column_info,
                        )
                    )

                    if existing_id is None:

                        inserted_count += 1

                        action = (
                            "INSERTED"
                        )

                    else:

                        updated_count += 1

                        action = (
                            "UPDATED"
                        )

                    print(
                        f"[DB] {action}"
                    )

                    print(
                        f"[DB] id = "
                        f"{db_id}"
                    )

                    results.append(
                        {
                            "status": (
                                action.lower()
                            ),

                            "db_id": (
                                db_id
                            ),

                            "source": (
                                source
                            ),

                            "source_id": (
                                source_id
                            ),

                            "topic_axis": (
                                document[
                                    "topic_axis"
                                ]
                            ),

                            "title": (
                                document[
                                    "title"
                                ]
                            ),

                            "content_hash": (
                                document[
                                    "content_hash"
                                ]
                            ),
                        }
                    )

                except Exception:

                    failed_count += 1
                    raise

            # =================================================
            # Verify inside transaction BEFORE commit
            # =================================================

            after_count = (
                _count_core_documents(
                    conn
                )
            )

            after_arxiv = (
                _count_source_documents(
                    conn,
                    "arxiv",
                )
            )

            after_ntrs = (
                _count_source_documents(
                    conn,
                    "ntrs",
                )
            )

            verification = (
                _verify_selected_documents(
                    conn,
                    documents,
                )
            )

            if verification[
                "verified"
            ] != EXPECTED_TOTAL:

                raise RuntimeError(
                    "Selected-document DB "
                    "verification failed.\n"
                    f"Verified: "
                    f"{verification['verified']}/"
                    f"{EXPECTED_TOTAL}\n"
                    f"Missing: "
                    f"{verification['missing']}\n"
                    f"Hash mismatch: "
                    f"{verification['hash_mismatch']}\n"
                    f"Axis mismatch: "
                    f"{verification['axis_mismatch']}"
                )

            # =================================================
            # First-run Gate
            # =================================================

            if (
                before_count
                == EXPECTED_FIRST_RUN_ROWS_BEFORE
                and
                before_arxiv
                == EXPECTED_FIRST_RUN_ARXIV_BEFORE
                and
                before_ntrs
                == EXPECTED_FIRST_RUN_NTRS_BEFORE
            ):

                if (
                    after_count
                    != EXPECTED_FIRST_RUN_ROWS_AFTER
                ):

                    raise RuntimeError(
                        "Unexpected total DB count.\n"
                        f"Expected "
                        f"{EXPECTED_FIRST_RUN_ROWS_AFTER}, "
                        f"found {after_count}."
                    )

                if (
                    after_arxiv
                    != EXPECTED_FIRST_RUN_ARXIV_AFTER
                ):

                    raise RuntimeError(
                        "Unexpected arXiv DB count.\n"
                        f"Expected "
                        f"{EXPECTED_FIRST_RUN_ARXIV_AFTER}, "
                        f"found {after_arxiv}."
                    )

                if (
                    after_ntrs
                    != EXPECTED_FIRST_RUN_NTRS_AFTER
                ):

                    raise RuntimeError(
                        "Unexpected NTRS DB count.\n"
                        f"Expected "
                        f"{EXPECTED_FIRST_RUN_NTRS_AFTER}, "
                        f"found {after_ntrs}."
                    )

            # =================================================
            # All checks passed -> COMMIT
            # =================================================

            conn.commit()

            # =================================================
            # Summary
            # =================================================

            _print_database_summary(
                conn
            )

            print()
            print("=" * 70)

            print(
                "ARXIV CORE-100 DB "
                "LOADING COMPLETED"
            )

            print("=" * 70)

            print(
                f"Local selected  : "
                f"{len(documents)}"
            )

            print(
                f"Inserted        : "
                f"{inserted_count}"
            )

            print(
                f"Updated         : "
                f"{updated_count}"
            )

            print(
                f"Duplicates      : "
                f"{duplicate_count}"
            )

            print(
                f"Failed          : "
                f"{failed_count}"
            )

            print()

            print(
                f"Rows before     : "
                f"{before_count}"
            )

            print(
                f"Rows after      : "
                f"{after_count}"
            )

            print(
                f"arXiv before    : "
                f"{before_arxiv}"
            )

            print(
                f"arXiv after     : "
                f"{after_arxiv}"
            )

            print(
                f"NTRS before     : "
                f"{before_ntrs}"
            )

            print(
                f"NTRS after      : "
                f"{after_ntrs}"
            )

            print()

            print(
                f"Verified        : "
                f"{verification['verified']} / "
                f"{EXPECTED_TOTAL}"
            )

            # =================================================
            # Report
            # =================================================

            report_payload = {
                "loader_version": (
                    LOADER_VERSION
                ),

                "selection_file": (
                    str(
                        SELECTION_FILE
                    )
                ),

                "local_selected": (
                    len(documents)
                ),

                "inserted": (
                    inserted_count
                ),

                "updated": (
                    updated_count
                ),

                "duplicates": (
                    duplicate_count
                ),

                "failed": (
                    failed_count
                ),

                "rows_before": (
                    before_count
                ),

                "rows_after": (
                    after_count
                ),

                "arxiv_before": (
                    before_arxiv
                ),

                "arxiv_after": (
                    after_arxiv
                ),

                "ntrs_before": (
                    before_ntrs
                ),

                "ntrs_after": (
                    after_ntrs
                ),

                "verification": (
                    verification
                ),

                "documents": (
                    results
                ),
            }

            _save_json(
                REPORT_FILE,
                report_payload,
            )

            print(
                f"Report           : "
                f"{REPORT_FILE}"
            )

            print("=" * 70)

            if (
                before_count
                == 30
                and
                inserted_count
                == 35
                and
                updated_count
                == 0
                and
                after_count
                == 65
                and
                after_arxiv
                == 50
                and
                after_ntrs
                == 15
                and
                verification[
                    "verified"
                ]
                == 35
            ):

                print(
                    "[PASS] Core-100 arXiv "
                    "addition completed."
                )

                print()

                print(
                    "AWS RDS:"
                )

                print(
                    "  arXiv : 50"
                )

                print(
                    "  NTRS  : 15"
                )

                print(
                    "  TOTAL : 65"
                )

                print()

                print(
                    "NEXT:"
                )

                print(
                    "Start NTRS Core-100 "
                    "expansion +35."
                )

            else:

                print(
                    "[PASS] Selected 35 arXiv "
                    "papers are present and "
                    "verified in DB."
                )

                print(
                    "This appears to be a rerun "
                    "or DB state differs from "
                    "the original 30-row state."
                )

            print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            "ARXIV CORE-100 "
            "DB LOADING FAILED"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "Transaction was not committed."
        )

        print(
            "The selected 35 documents "
            "were not partially loaded."
        )

        print("=" * 70)

        raise


if __name__ == "__main__":
    main()