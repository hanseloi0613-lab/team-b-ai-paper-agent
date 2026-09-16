import json
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - NASA NTRS Core-100 DB Loader
# ============================================================
#
# Current AWS RDS:
#
#   arXiv : 50
#   NTRS  : 15
#   TOTAL : 65
#
#
# Add selected NTRS:
#
#   rover_autonomy       +12
#   onboard_ai           +12
#   satellite_autonomy   +11
#   ------------------------
#   TOTAL                +35
#
#
# Final Core-100:
#
#   arXiv : 50
#   NTRS  : 50
#   TOTAL : 100
#
#
# Final axis distribution:
#
#   rover_autonomy       34
#   onboard_ai           33
#   satellite_autonomy   33
#
#
# Safety:
#
# - frozen ntrs_core100_selected.json ONLY
# - quality.passed == True required
# - local SHA-256 duplicate check
# - DB content_hash duplicate check
# - schema validation
# - full transaction
# - verify before COMMIT
# - no partial insert
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
    "onboard_ai": 12,
    "satellite_autonomy": 11,
}

EXPECTED_TOTAL = sum(
    EXPECTED_COUNTS.values()
)


# ============================================================
# Expected DB State
# ============================================================

EXPECTED_FIRST_RUN = {
    "total": 65,
    "arxiv": 50,
    "ntrs": 15,
}

EXPECTED_FINAL = {
    "total": 100,
    "arxiv": 50,
    "ntrs": 50,
}


EXPECTED_FINAL_SOURCE_AXIS = {
    "arxiv": {
        "rover_autonomy": 17,
        "onboard_ai": 16,
        "satellite_autonomy": 17,
    },
    "ntrs": {
        "rover_autonomy": 17,
        "onboard_ai": 17,
        "satellite_autonomy": 16,
    },
}


EXPECTED_FINAL_GLOBAL_AXIS = {
    "rover_autonomy": 34,
    "onboard_ai": 33,
    "satellite_autonomy": 33,
}


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
    / "ntrs_core100_db_loading_report.json"
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
# Selection Manifest
# ============================================================

def _load_selection() -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, dict]],
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
        and version != LOADER_VERSION
    ):

        raise RuntimeError(
            "Selection version mismatch.\n"
            f"Expected: {LOADER_VERSION}\n"
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
    ).strip().lower()

    if (
        source
        and source != "ntrs"
    ):

        raise RuntimeError(
            f"Unexpected selection source: "
            f"{source}"
        )

    # --------------------------------------------------------
    # Canonical ID list
    # --------------------------------------------------------

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
            "selected_documents missing."
        )

    # --------------------------------------------------------
    # Frozen metadata snapshots
    # --------------------------------------------------------

    selected_records = (
        payload.get(
            "selected_records"
        )
    )

    if not isinstance(
        selected_records,
        dict,
    ):

        raise RuntimeError(
            "selected_records missing."
        )

    selection = {}

    metadata_index = {}

    global_ids = set()

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        ids = (
            selected_documents.get(
                topic_axis
            )
        )

        records = (
            selected_records.get(
                topic_axis
            )
        )

        if not isinstance(
            ids,
            list,
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                "selected ID list missing."
            )

        if not isinstance(
            records,
            list,
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                "selected metadata records missing."
            )

        if len(ids) != expected_count:

            raise RuntimeError(
                f"{topic_axis}: expected "
                f"{expected_count}, "
                f"found {len(ids)}."
            )

        if len(records) != expected_count:

            raise RuntimeError(
                f"{topic_axis}: metadata expected "
                f"{expected_count}, "
                f"found {len(records)}."
            )

        record_index = {}

        for record in records:

            if not isinstance(
                record,
                dict,
            ):

                raise RuntimeError(
                    f"{topic_axis}: invalid metadata."
                )

            source_id = str(
                record.get(
                    "source_id",
                    "",
                )
            ).strip()

            if not source_id:

                raise RuntimeError(
                    f"{topic_axis}: metadata source_id missing."
                )

            record_index[
                source_id
            ] = record

        normalized_ids = []

        for raw_source_id in ids:

            source_id = str(
                raw_source_id
            ).strip()

            if not source_id:

                raise RuntimeError(
                    f"{topic_axis}: empty source_id."
                )

            if source_id in global_ids:

                raise RuntimeError(
                    "Cross-axis duplicate NTRS ID: "
                    f"{source_id}"
                )

            global_ids.add(
                source_id
            )

            if source_id not in record_index:

                raise RuntimeError(
                    f"{source_id}: "
                    "metadata snapshot missing."
                )

            metadata = (
                record_index[
                    source_id
                ]
            )

            metadata_axis = str(
                metadata.get(
                    "topic_axis",
                    "",
                )
            )

            if (
                metadata_axis
                and metadata_axis
                != topic_axis
            ):

                raise RuntimeError(
                    f"{source_id}: "
                    f"metadata axis mismatch: "
                    f"{metadata_axis}"
                )

            normalized_ids.append(
                source_id
            )

            metadata_index[
                source_id
            ] = metadata

        selection[
            topic_axis
        ] = normalized_ids

    total = sum(
        len(
            source_ids
        )
        for source_ids
        in selection.values()
    )

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} selected "
            f"NTRS papers, found {total}."
        )

    return (
        selection,
        metadata_index,
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
# Cleaned Document Paths
# ============================================================

def _find_cleaned_documents(
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

    documents = []

    errors = []

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

            cleaned_path = (
                paper_dir
                / "cleaned_document.json"
            )

            raw_path = (
                paper_dir
                / "raw_content.txt"
            )

            clean_path = (
                paper_dir
                / "clean_content.txt"
            )

            resolution_path = (
                paper_dir
                / "resolution.json"
            )

            for path in (
                cleaned_path,
                raw_path,
                clean_path,
                resolution_path,
            ):

                if not path.exists():

                    errors.append(
                        (
                            topic_axis,
                            source_id,
                            f"missing: {path.name}",
                        )
                    )

            if (
                cleaned_path.exists()
                and raw_path.exists()
                and clean_path.exists()
                and resolution_path.exists()
            ):

                documents.append(
                    (
                        topic_axis,
                        source_id,
                        cleaned_path,
                    )
                )

    if errors:

        print()

        print(
            "Local preflight errors:"
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
            f"{len(errors)} local "
            "file error(s)."
        )

    if (
        len(documents)
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} "
            f"cleaned documents, "
            f"found {len(documents)}."
        )

    return documents


# ============================================================
# PostgreSQL Schema
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
            f"{TABLE_SCHEMA}.{TABLE_NAME}"
        )

    result = {}

    for (
        column_name,
        data_type,
        udt_name,
        is_nullable,
        column_default,
    ) in rows:

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

    missing = (
        REQUIRED_COLUMNS
        - set(
            column_info.keys()
        )
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
# Parser Name
# ============================================================

def _parser_name(
    source_info: dict,
) -> str:

    parser_mode = str(
        source_info.get(
            "parser_mode",
            "",
        )
    )

    if parser_mode:

        return parser_mode

    selected_format = str(
        source_info.get(
            "selected_format",
            "",
        )
    ).lower()

    if selected_format in {
        "txt",
        "original_text",
    }:

        return "ntrs_txt"

    if selected_format == "pdf":

        return "ntrs_pdf_pypdf"

    return "ntrs_parser_unknown"


# ============================================================
# Build One DB Document
# ============================================================

def _build_document(
    *,
    topic_axis: str,
    source_id: str,
    cleaned_path: Path,
    metadata: dict,
) -> dict:

    paper_dir = (
        cleaned_path.parent
    )

    cleaned = (
        _load_json(
            cleaned_path
        )
    )

    resolution = (
        _load_json(
            paper_dir
            / "resolution.json"
        )
    )

    parsed = (
        _load_json(
            paper_dir
            / "parsed_document.json"
        )
    )

    raw_content = (
        (
            paper_dir
            / "raw_content.txt"
        )
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    clean_content = (
        (
            paper_dir
            / "clean_content.txt"
        )
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    # ========================================================
    # Identity
    # ========================================================

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
            f"{source_id}: cleaned source_id "
            f"mismatch: {cleaned_source_id}"
        )

    if (
        cleaned_axis
        != topic_axis
    ):

        raise RuntimeError(
            f"{source_id}: cleaned axis "
            f"mismatch: {cleaned_axis}"
        )

    # ========================================================
    # Quality
    # ========================================================

    quality = (
        cleaned.get(
            "quality",
            {},
        )
    )

    if not isinstance(
        quality,
        dict,
    ):

        raise RuntimeError(
            f"{source_id}: invalid quality."
        )

    if not quality.get(
        "passed",
        False,
    ):

        raise RuntimeError(
            f"{source_id}: "
            "quality.passed != True"
        )

    # ========================================================
    # Content
    # ========================================================

    if not raw_content.strip():

        raise RuntimeError(
            f"{source_id}: "
            "raw_content empty."
        )

    if not clean_content.strip():

        raise RuntimeError(
            f"{source_id}: "
            "clean_content empty."
        )

    content_hash = str(
        cleaned.get(
            "content_hash",
            "",
        )
    ).strip()

    if not content_hash:

        raise RuntimeError(
            f"{source_id}: "
            "content_hash missing."
        )

    # ========================================================
    # Hash Verification
    #
    # cleaned_document hash must match actual clean_content.txt
    # ========================================================

    import hashlib

    actual_hash = (
        hashlib.sha256(
            clean_content.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    if actual_hash != content_hash:

        raise RuntimeError(
            f"{source_id}: "
            "clean_content SHA-256 mismatch."
        )

    # ========================================================
    # Metadata
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

        "query_scores": (
            metadata.get(
                "query_scores",
                {},
            )
        ),

        "api_score_max": (
            metadata.get(
                "api_score_max"
            )
        ),

        "distribution": (
            metadata.get(
                "distribution"
            )
        ),

        "disseminated": (
            metadata.get(
                "disseminated"
            )
        ),

        "downloads_available": (
            metadata.get(
                "downloads_available"
            )
        ),

        "fulltext_url": (
            metadata.get(
                "fulltext_url"
            )
        ),

        "original_url": (
            metadata.get(
                "original_url"
            )
        ),

        "downloads": (
            metadata.get(
                "downloads",
                [],
            )
        ),

        "selection": (
            metadata.get(
                "selection",
                {}
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

        "statistics": (
            cleaned.get(
                "statistics",
                {},
            )
        ),

        "cleaning": (
            cleaned.get(
                "cleaning",
                {},
            )
        ),

        "pipeline": {
            "selection": (
                "ntrs_core100_selected"
            ),
            "parser": (
                _parser_name(
                    source_info
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

    # ========================================================
    # Final DB Record
    # ========================================================

    return {
        "source": (
            "ntrs"
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
            or "NTRS_DOCUMENT"
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
            or "v1"
        ),

        "char_count": (
            len(
                clean_content
            )
        ),

        "parse_status": (
            "success"
        ),

        "metadata": (
            db_metadata
        ),
    }


# ============================================================
# Local Documents
# ============================================================

def _load_local_documents(
    selection: dict[
        str,
        list[str],
    ],
    metadata_index: dict[
        str,
        dict,
    ],
) -> list[dict]:

    paths = (
        _find_cleaned_documents(
            selection
        )
    )

    documents = []

    print(
        f"[LOCAL] Selected cleaned documents: "
        f"{len(paths)}"
    )

    for (
        index,
        (
            topic_axis,
            source_id,
            cleaned_path,
        ),
    ) in enumerate(
        paths,
        start=1,
    ):

        metadata = (
            metadata_index.get(
                source_id
            )
        )

        if metadata is None:

            raise RuntimeError(
                f"{source_id}: "
                "metadata snapshot missing."
            )

        document = (
            _build_document(
                topic_axis=(
                    topic_axis
                ),
                source_id=(
                    source_id
                ),
                cleaned_path=(
                    cleaned_path
                ),
                metadata=(
                    metadata
                ),
            )
        )

        documents.append(
            document
        )

        print(
            f"[LOCAL {index:02d}/"
            f"{EXPECTED_TOTAL}] "
            f"{topic_axis} | "
            f"{source_id} | "
            f"{document['char_count']} chars"
        )

    if (
        len(documents)
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} "
            f"DB documents, "
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

        digest = (
            document[
                "content_hash"
            ]
        )

        source_id = (
            document[
                "source_id"
            ]
        )

        if digest in seen:

            raise RuntimeError(
                "Local duplicate content hash:\n"
                f"{seen[digest]} "
                f"<-> {source_id}"
            )

        seen[
            digest
        ] = source_id


# ============================================================
# Counts
# ============================================================

def _count_total(
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


def _count_source(
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


def _source_axis_counts(
    conn: psycopg.Connection,
) -> dict[
    str,
    dict[
        str,
        int,
    ],
]:

    sql = f"""
    SELECT
        source,
        topic_axis,
        COUNT(*)
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    GROUP BY
        source,
        topic_axis
    ORDER BY
        source,
        topic_axis
    """

    result = {}

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

        result.setdefault(
            str(
                source
            ),
            {},
        )[
            str(
                topic_axis
            )
        ] = int(
            count
        )

    return result


def _global_axis_counts(
    conn: psycopg.Connection,
) -> dict[
    str,
    int,
]:

    sql = f"""
    SELECT
        topic_axis,
        COUNT(*)
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    GROUP BY topic_axis
    ORDER BY topic_axis
    """

    result = {}

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = (
            cur.fetchall()
        )

    for (
        topic_axis,
        count,
    ) in rows:

        result[
            str(
                topic_axis
            )
        ] = int(
            count
        )

    return result


# ============================================================
# Existing Documents
# ============================================================

def _find_existing_source(
    conn: psycopg.Connection,
    source: str,
    source_id: str,
) -> tuple[
    int,
    str,
] | None:

    sql = f"""
    SELECT
        id,
        content_hash
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

    return (
        int(
            row[0]
        ),
        str(
            row[1]
        ),
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
# DB Preflight
# ============================================================

def _database_preflight(
    conn: psycopg.Connection,
    documents: list[dict],
) -> dict:

    total = (
        _count_total(
            conn
        )
    )

    arxiv = (
        _count_source(
            conn,
            "arxiv",
        )
    )

    ntrs = (
        _count_source(
            conn,
            "ntrs",
        )
    )

    print()
    print(
        f"[DB] Rows before  : "
        f"{total}"
    )

    print(
        f"[DB] arXiv before : "
        f"{arxiv}"
    )

    print(
        f"[DB] NTRS before  : "
        f"{ntrs}"
    )

    first_run = (
        total
        == EXPECTED_FIRST_RUN[
            "total"
        ]
        and arxiv
        == EXPECTED_FIRST_RUN[
            "arxiv"
        ]
        and ntrs
        == EXPECTED_FIRST_RUN[
            "ntrs"
        ]
    )

    completed_rerun = (
        total
        == EXPECTED_FINAL[
            "total"
        ]
        and arxiv
        == EXPECTED_FINAL[
            "arxiv"
        ]
        and ntrs
        == EXPECTED_FINAL[
            "ntrs"
        ]
    )

    if not (
        first_run
        or completed_rerun
    ):

        raise RuntimeError(
            "Unexpected AWS DB state.\n"
            "Expected either:\n"
            "  first run : total=65, "
            "arxiv=50, ntrs=15\n"
            "or\n"
            "  rerun     : total=100, "
            "arxiv=50, ntrs=50\n"
            f"Actual      : total={total}, "
            f"arxiv={arxiv}, ntrs={ntrs}"
        )

    if first_run:

        print(
            "[DB] Core-100 first-run "
            "state detected."
        )

    else:

        print(
            "[DB] Completed Core-100 "
            "rerun state detected."
        )

    existing_selected = 0

    # ========================================================
    # Verify ALL selected docs before mutation
    # ========================================================

    for document in documents:

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

        content_hash = (
            document[
                "content_hash"
            ]
        )

        existing = (
            _find_existing_source(
                conn,
                source,
                source_id,
            )
        )

        hash_owner = (
            _find_hash_owner(
                conn,
                content_hash,
            )
        )

        # ----------------------------------------------------
        # Existing source ID
        # ----------------------------------------------------

        if existing is not None:

            existing_selected += 1

            (
                _existing_id,
                existing_hash,
            ) = existing

            if (
                existing_hash
                != content_hash
            ):

                raise RuntimeError(
                    "Existing source_id has "
                    "different content hash:\n"
                    f"{source}/"
                    f"{source_id}"
                )

        # ----------------------------------------------------
        # Hash belongs to another document
        # ----------------------------------------------------

        if hash_owner is not None:

            (
                owner_id,
                owner_source,
                owner_source_id,
            ) = hash_owner

            same_document = (
                owner_source
                == source
                and owner_source_id
                == source_id
            )

            if not same_document:

                raise RuntimeError(
                    "DB content-hash collision:\n"
                    f"Current  : "
                    f"{source}/{source_id}\n"
                    f"Existing : "
                    f"{owner_source}/"
                    f"{owner_source_id}\n"
                    f"DB id    : "
                    f"{owner_id}"
                )

    if first_run:

        if existing_selected != 0:

            raise RuntimeError(
                "First-run state but some "
                "selected NTRS IDs already exist: "
                f"{existing_selected}"
            )

    if completed_rerun:

        if (
            existing_selected
            != EXPECTED_TOTAL
        ):

            raise RuntimeError(
                "Completed DB state detected "
                "but not all selected NTRS "
                "documents exist."
            )

    print(
        "[DB] Preflight selected source IDs "
        "and hashes: PASS"
    )

    return {
        "total": (
            total
        ),
        "arxiv": (
            arxiv
        ),
        "ntrs": (
            ntrs
        ),
        "first_run": (
            first_run
        ),
    }


# ============================================================
# UPSERT
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
            "UPSERT did not return DB id: "
            f"{document['source_id']}"
        )

    return int(
        row[0]
    )


# ============================================================
# Verify Selected 35
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

    verified = (
        len(
            documents
        )
        - len(
            missing
        )
        - len(
            hash_mismatch
        )
        - len(
            axis_mismatch
        )
    )

    return {
        "total": (
            len(
                documents
            )
        ),
        "verified": (
            verified
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
# Final Core-100 Distribution Gate
# ============================================================

def _validate_final_distribution(
    conn: psycopg.Connection,
) -> dict:

    total = (
        _count_total(
            conn
        )
    )

    arxiv = (
        _count_source(
            conn,
            "arxiv",
        )
    )

    ntrs = (
        _count_source(
            conn,
            "ntrs",
        )
    )

    source_axis = (
        _source_axis_counts(
            conn
        )
    )

    global_axis = (
        _global_axis_counts(
            conn
        )
    )

    if total != 100:

        raise RuntimeError(
            f"Final total expected 100, "
            f"found {total}."
        )

    if arxiv != 50:

        raise RuntimeError(
            f"Final arXiv expected 50, "
            f"found {arxiv}."
        )

    if ntrs != 50:

        raise RuntimeError(
            f"Final NTRS expected 50, "
            f"found {ntrs}."
        )

    # ========================================================
    # Source × Axis
    # ========================================================

    for (
        source,
        expected_axes,
    ) in (
        EXPECTED_FINAL_SOURCE_AXIS.items()
    ):

        actual_axes = (
            source_axis.get(
                source,
                {},
            )
        )

        for (
            topic_axis,
            expected_count,
        ) in expected_axes.items():

            actual_count = (
                actual_axes.get(
                    topic_axis,
                    0,
                )
            )

            if (
                actual_count
                != expected_count
            ):

                raise RuntimeError(
                    "Source/axis distribution "
                    "mismatch:\n"
                    f"{source} / "
                    f"{topic_axis}\n"
                    f"Expected: "
                    f"{expected_count}\n"
                    f"Actual  : "
                    f"{actual_count}"
                )

    # ========================================================
    # Global Axis
    # ========================================================

    for (
        topic_axis,
        expected_count,
    ) in (
        EXPECTED_FINAL_GLOBAL_AXIS.items()
    ):

        actual_count = (
            global_axis.get(
                topic_axis,
                0,
            )
        )

        if (
            actual_count
            != expected_count
        ):

            raise RuntimeError(
                "Global axis distribution "
                "mismatch:\n"
                f"{topic_axis}: "
                f"expected "
                f"{expected_count}, "
                f"found "
                f"{actual_count}"
            )

    return {
        "total": (
            total
        ),
        "arxiv": (
            arxiv
        ),
        "ntrs": (
            ntrs
        ),
        "source_axis": (
            source_axis
        ),
        "global_axis": (
            global_axis
        ),
    }


# ============================================================
# Print DB Summary
# ============================================================

def _print_database_summary(
    final_state: dict,
) -> None:

    print()
    print("=" * 78)

    print(
        "FINAL CORE-100 DATABASE SUMMARY"
    )

    print("=" * 78)

    source_axis = (
        final_state[
            "source_axis"
        ]
    )

    for source in (
        "arxiv",
        "ntrs",
    ):

        axes = (
            source_axis.get(
                source,
                {},
            )
        )

        for topic_axis in (
            "rover_autonomy",
            "onboard_ai",
            "satellite_autonomy",
        ):

            print(
                f"{source:10} | "
                f"{topic_axis:22} | "
                f"{axes.get(topic_axis, 0)}"
            )

    print("-" * 78)

    print(
        f"TOTAL documents          : "
        f"{final_state['total']}"
    )

    print(
        f"arXiv documents          : "
        f"{final_state['arxiv']}"
    )

    print(
        f"NTRS documents           : "
        f"{final_state['ntrs']}"
    )

    print()

    print(
        "Global axis totals:"
    )

    for topic_axis in (
        "rover_autonomy",
        "onboard_ai",
        "satellite_autonomy",
    ):

        print(
            f"  {topic_axis:22} : "
            f"{final_state['global_axis'].get(topic_axis, 0)}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - NASA NTRS "
        "Core-100 DB Loader"
    )

    print("=" * 78)

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
        f"Report         : "
        f"{REPORT_FILE}"
    )

    # ========================================================
    # Local Preflight
    # ========================================================

    (
        selection,
        metadata_index,
    ) = (
        _load_selection()
    )

    print()
    print(
        "Frozen NTRS selection:"
    )

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        print(
            f"  {topic_axis:22} : "
            f"{len(selection[topic_axis])} / "
            f"{expected_count}"
        )

    print(
        f"  {'TOTAL':22} : "
        f"{sum(len(x) for x in selection.values())}"
    )

    documents = (
        _load_local_documents(
            selection,
            metadata_index,
        )
    )

    _validate_local_hashes(
        documents
    )

    print()
    print(
        "[LOCAL] All 35 selected NTRS "
        "documents passed preflight."
    )

    # ========================================================
    # AWS
    # ========================================================

    inserted_count = 0

    updated_count = 0

    failed_count = 0

    results = []

    print()
    print(
        "[DB] Connecting to AWS RDS..."
    )

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
            # Full DB Preflight BEFORE mutation
            # =================================================

            before_state = (
                _database_preflight(
                    conn,
                    documents,
                )
            )

            # =================================================
            # UPSERT
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

                existing = (
                    _find_existing_source(
                        conn,
                        source,
                        source_id,
                    )
                )

                print()
                print("-" * 78)

                print(
                    f"[{index}/"
                    f"{EXPECTED_TOTAL}] "
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

                    db_id = (
                        _upsert_document(
                            conn,
                            document,
                            column_info,
                        )
                    )

                    if existing is None:

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
            # Selected 35 Verification
            # =================================================

            verification = (
                _verify_selected_documents(
                    conn,
                    documents,
                )
            )

            if (
                verification[
                    "verified"
                ]
                != EXPECTED_TOTAL
            ):

                raise RuntimeError(
                    "Selected-document verification "
                    "failed.\n"
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
            # Core-100 Final Distribution
            # =================================================

            final_state = (
                _validate_final_distribution(
                    conn
                )
            )

            # =================================================
            # All Gates PASS -> COMMIT
            # =================================================

            conn.commit()

            # =================================================
            # Console Summary
            # =================================================

            _print_database_summary(
                final_state
            )

            print()
            print("=" * 78)

            print(
                "NTRS CORE-100 "
                "DB LOADING COMPLETED"
            )

            print("=" * 78)

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
                f"Failed          : "
                f"{failed_count}"
            )

            print()

            print(
                f"Rows before     : "
                f"{before_state['total']}"
            )

            print(
                f"Rows after      : "
                f"{final_state['total']}"
            )

            print(
                f"arXiv before    : "
                f"{before_state['arxiv']}"
            )

            print(
                f"arXiv after     : "
                f"{final_state['arxiv']}"
            )

            print(
                f"NTRS before     : "
                f"{before_state['ntrs']}"
            )

            print(
                f"NTRS after      : "
                f"{final_state['ntrs']}"
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

            report = {
                "loader_version": (
                    LOADER_VERSION
                ),

                "selection_file": (
                    str(
                        SELECTION_FILE
                    )
                ),

                "local_selected": (
                    len(
                        documents
                    )
                ),

                "inserted": (
                    inserted_count
                ),

                "updated": (
                    updated_count
                ),

                "failed": (
                    failed_count
                ),

                "before_state": (
                    before_state
                ),

                "final_state": (
                    final_state
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
                report,
            )

            print(
                f"Report          : "
                f"{REPORT_FILE}"
            )

            print("=" * 78)

            # =================================================
            # First Run Success
            # =================================================

            if (
                before_state[
                    "first_run"
                ]
                and inserted_count
                == EXPECTED_TOTAL
                and updated_count
                == 0
            ):

                print(
                    "[PASS] CORE-100 COMPLETE."
                )

                print()

                print(
                    "AWS RDS:"
                )

                print(
                    "  arXiv : 50"
                )

                print(
                    "  NTRS  : 50"
                )

                print(
                    "  TOTAL : 100"
                )

                print()

                print(
                    "Axis totals:"
                )

                print(
                    "  rover_autonomy       : 34"
                )

                print(
                    "  onboard_ai           : 33"
                )

                print(
                    "  satellite_autonomy   : 33"
                )

                print()

                print(
                    "NEXT:"
                )

                print(
                    "Freeze Core-100 and "
                    "start RAG chunking."
                )

            else:

                print(
                    "[PASS] Existing Core-100 "
                    "documents verified and refreshed."
                )

            print("=" * 78)

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "NTRS CORE-100 "
            "DB LOADING FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "Transaction was not committed."
        )

        print(
            "No partial NTRS Core-100 "
            "load was preserved."
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()