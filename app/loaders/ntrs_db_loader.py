import json
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from app.config import PROJECT_ROOT, settings


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
    / "ntrs_db_loading_report.json"
)


# ============================================================
# Database
# ============================================================

DB_SCHEMA = "public"
DB_TABLE = "core_documents"


# ============================================================
# Expected Core Columns
# ============================================================
#
# 이 loader는 현재 프로젝트의 canonical core_documents
# schema를 대상으로 한다.
#
# information_schema를 읽어서 실제 DB column type도
# 확인하므로 authors/categories/metadata가
# JSONB / ARRAY / TEXT 중 무엇인지에 맞춰 값을 변환한다.
# ============================================================

REQUIRED_COLUMNS = {
    "source",
    "source_id",
    "title",
    "topic_axis",
    "language",
    "raw_content",
    "clean_content",
    "content_hash",
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
# Text Helper
# ============================================================

def _read_text(
    path: Path,
) -> str:

    if not path.exists():

        raise FileNotFoundError(
            f"Text file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()


# ============================================================
# Date Helper
# ============================================================

def _parse_date(
    value: Any,
) -> date | None:

    if value is None:

        return None

    if isinstance(
        value,
        date,
    ):

        return value

    text = str(
        value
    ).strip()

    if not text:

        return None

    # 2024-01-01T00:00:00Z
    #               ↓
    # 2024-01-01
    try:

        return date.fromisoformat(
            text[:10]
        )

    except ValueError:

        print(
            f"[WARN] Invalid date ignored: "
            f"{value}"
        )

        return None


# ============================================================
# Clean Local Path From Provenance
# ============================================================

def _clean_resolution_for_db(
    resolution: dict,
) -> dict:
    """
    resolution.json에는 로컬 Windows 경로가 들어갈 수 있다.

    AWS DB provenance에:

        E:\\자료실\\...

    같은 개발 PC 경로까지 넣을 필요는 없다.

    따라서 source URL / selected format / quality 등은
    보존하고 local_file 계열만 제거한다.
    """

    cleaned = dict(
        resolution
    )

    cleaned.pop(
        "local_file",
        None,
    )

    cleaned.pop(
        "original_file",
        None,
    )

    return cleaned


# ============================================================
# Find Local Documents
# ============================================================

def _find_cleaned_documents() -> list[Path]:

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            f"NTRS resolved root not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/cleaned_document.json"
        )
    )


# ============================================================
# Build One Canonical DB Document
# ============================================================

def _build_document(
    cleaned_path: Path,
) -> dict:

    paper_dir = (
        cleaned_path.parent
    )

    # ========================================================
    # Required Files
    # ========================================================

    metadata_path = (
        paper_dir
        / "metadata.json"
    )

    resolution_path = (
        paper_dir
        / "resolution.json"
    )

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    clean_path = (
        paper_dir
        / "clean_content.txt"
    )

    # ========================================================
    # Load
    # ========================================================

    cleaned = (
        _load_json(
            cleaned_path
        )
    )

    metadata = (
        _load_json(
            metadata_path
        )
    )

    resolution = (
        _load_json(
            resolution_path
        )
    )

    raw_content = (
        _read_text(
            raw_path
        )
    )

    clean_content = (
        _read_text(
            clean_path
        )
    )

    # ========================================================
    # Quality Gate
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
            f"Invalid quality object: "
            f"{cleaned_path}"
        )

    if not quality.get(
        "passed",
        False,
    ):

        raise RuntimeError(
            "Quality check did not pass: "
            f"{paper_dir}"
        )

    # ========================================================
    # Hash Gate
    # ========================================================

    content_hash = (
        cleaned.get(
            "content_hash"
        )
    )

    if not content_hash:

        raise RuntimeError(
            f"content_hash missing: "
            f"{cleaned_path}"
        )

    content_hash = str(
        content_hash
    ).strip()

    if len(
        content_hash
    ) != 64:

        raise RuntimeError(
            f"Invalid SHA-256 hash: "
            f"{content_hash}"
        )

    # ========================================================
    # Canonical Identity
    # ========================================================

    source_id = str(
        cleaned.get(
            "source_id"
        )
        or metadata.get(
            "source_id"
        )
        or paper_dir.name
    ).strip()

    topic_axis = str(
        cleaned.get(
            "topic_axis"
        )
        or metadata.get(
            "topic_axis"
        )
        or paper_dir.parent.name
    ).strip()

    title = str(
        cleaned.get(
            "title"
        )
        or metadata.get(
            "title"
        )
        or ""
    ).strip()

    abstract = str(
        cleaned.get(
            "abstract"
        )
        or metadata.get(
            "abstract"
        )
        or ""
    ).strip()

    if not source_id:

        raise RuntimeError(
            f"source_id missing: "
            f"{paper_dir}"
        )

    if not title:

        raise RuntimeError(
            f"title missing: "
            f"{paper_dir}"
        )

    if topic_axis not in {
        "rover_autonomy",
        "onboard_ai",
        "satellite_autonomy",
    }:

        raise RuntimeError(
            f"Invalid topic_axis: "
            f"{topic_axis}"
        )

    # ========================================================
    # Metadata
    # ========================================================

    authors = (
        metadata.get(
            "authors",
            [],
        )
    )

    if not isinstance(
        authors,
        list,
    ):

        authors = [
            str(
                authors
            )
        ]

    categories = (
        metadata.get(
            "categories",
            [],
        )
    )

    if not isinstance(
        categories,
        list,
    ):

        categories = [
            str(
                categories
            )
        ]

    # ========================================================
    # Provenance
    # ========================================================

    collector_metadata = (
        metadata.get(
            "metadata",
            {},
        )
    )

    if not isinstance(
        collector_metadata,
        dict,
    ):

        collector_metadata = {
            "raw_value": (
                collector_metadata
            )
        }

    provenance = {
        "provider": (
            "NASA NTRS"
        ),

        "collection": {
            "matched_queries": (
                metadata.get(
                    "matched_queries",
                    [],
                )
            ),

            "api_score": (
                metadata.get(
                    "api_score"
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

            "only_abstract": (
                metadata.get(
                    "only_abstract"
                )
            ),

            "document_type_details": (
                metadata.get(
                    "document_type_details"
                )
            ),
        },

        "downloads": {
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

            "pdf_url": (
                metadata.get(
                    "pdf_url"
                )
            ),

            "items": (
                metadata.get(
                    "downloads",
                    [],
                )
            ),
        },

        "content_resolution": (
            _clean_resolution_for_db(
                resolution
            )
        ),

        "cleaning": (
            cleaned.get(
                "cleaning",
                {},
            )
        ),

        "quality": (
            quality
        ),

        "statistics": (
            cleaned.get(
                "statistics",
                {},
            )
        ),

        "source_metadata": (
            collector_metadata
        ),
    }

    # ========================================================
    # Canonical Row
    # ========================================================

    return {
        "source": "ntrs",

        "source_id": (
            source_id
        ),

        "title": (
            title
        ),

        "abstract": (
            abstract
        ),

        "authors": (
            authors
        ),

        "categories": (
            categories
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

        "char_count": len(
            clean_content
        ),

        "parse_status": (
            "success"
        ),

        "metadata": (
            provenance
        ),
    }


# ============================================================
# Local Validation
# ============================================================

def _prepare_local_documents(
    paths: list[Path],
) -> list[dict]:

    documents: list[
        dict
    ] = []

    errors = []

    seen_source_ids = set()
    seen_hashes = set()

    for path in paths:

        try:

            document = (
                _build_document(
                    path
                )
            )

        except Exception as exc:

            errors.append(
                {
                    "path": str(
                        path
                    ),

                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

            continue

        identity = (
            document[
                "source"
            ],
            document[
                "source_id"
            ],
        )

        if (
            identity
            in seen_source_ids
        ):

            errors.append(
                {
                    "path": str(
                        path
                    ),

                    "error": (
                        "Duplicate local "
                        "(source, source_id): "
                        f"{identity}"
                    ),
                }
            )

            continue

        seen_source_ids.add(
            identity
        )

        digest = (
            document[
                "content_hash"
            ]
        )

        if (
            digest
            in seen_hashes
        ):

            errors.append(
                {
                    "path": str(
                        path
                    ),

                    "error": (
                        "Duplicate local "
                        f"content_hash: {digest}"
                    ),
                }
            )

            continue

        seen_hashes.add(
            digest
        )

        documents.append(
            document
        )

    # ========================================================
    # All-or-nothing local gate
    # ========================================================

    if errors:

        print()
        print("=" * 70)

        print(
            "LOCAL VALIDATION FAILED"
        )

        print("=" * 70)

        for item in errors:

            print(
                f"[ERROR] "
                f"{item['path']}"
            )

            print(
                f"        "
                f"{item['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} local "
            "document(s) failed validation."
        )

    return documents


# ============================================================
# Schema Inspection
# ============================================================

def _get_table_columns(
    cursor,
) -> dict[str, dict]:

    cursor.execute(
        """
        SELECT
            column_name,
            data_type,
            udt_name,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (
            DB_SCHEMA,
            DB_TABLE,
        ),
    )

    rows = (
        cursor.fetchall()
    )

    if not rows:

        raise RuntimeError(
            f"Table not found: "
            f"{DB_SCHEMA}.{DB_TABLE}"
        )

    columns = {}

    for (
        column_name,
        data_type,
        udt_name,
        is_nullable,
    ) in rows:

        columns[
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
        }

    return columns


# ============================================================
# Validate Schema
# ============================================================

def _validate_schema(
    columns: dict[str, dict],
) -> None:

    missing = (
        REQUIRED_COLUMNS
        - set(
            columns.keys()
        )
    )

    if missing:

        raise RuntimeError(
            "core_documents is missing "
            "required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


# ============================================================
# Adapt Python Value To Actual PostgreSQL Column
# ============================================================

def _adapt_value(
    value: Any,
    column_info: dict,
) -> Any:

    if value is None:

        return None

    data_type = str(
        column_info.get(
            "data_type",
            "",
        )
    ).lower()

    udt_name = str(
        column_info.get(
            "udt_name",
            "",
        )
    ).lower()

    # ========================================================
    # JSON / JSONB
    # ========================================================

    if (
        data_type
        in {
            "json",
            "jsonb",
        }
        or udt_name
        in {
            "json",
            "jsonb",
        }
    ):

        return Jsonb(
            value
        )

    # ========================================================
    # PostgreSQL ARRAY
    # ========================================================

    if (
        data_type
        == "array"
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
            value
        ]

    # ========================================================
    # Text column but Python value is list/dict
    # ========================================================

    if isinstance(
        value,
        (
            list,
            dict,
        ),
    ):

        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return value


# ============================================================
# Count Rows
# ============================================================

def _count_rows(
    cursor,
) -> int:

    cursor.execute(
        sql.SQL(
            "SELECT COUNT(*) "
            "FROM {}.{}"
        ).format(
            sql.Identifier(
                DB_SCHEMA
            ),
            sql.Identifier(
                DB_TABLE
            ),
        )
    )

    return int(
        cursor.fetchone()[0]
    )


# ============================================================
# Find Existing By Source Identity
# ============================================================

def _find_existing_source(
    cursor,
    source: str,
    source_id: str,
) -> dict | None:

    cursor.execute(
        sql.SQL(
            """
            SELECT
                id,
                source,
                source_id,
                content_hash
            FROM {}.{}
            WHERE source = %s
              AND source_id = %s
            LIMIT 1
            """
        ).format(
            sql.Identifier(
                DB_SCHEMA
            ),
            sql.Identifier(
                DB_TABLE
            ),
        ),
        (
            source,
            source_id,
        ),
    )

    row = (
        cursor.fetchone()
    )

    if row is None:

        return None

    return {
        "id": (
            row[0]
        ),

        "source": (
            row[1]
        ),

        "source_id": (
            row[2]
        ),

        "content_hash": (
            row[3]
        ),
    }


# ============================================================
# Find Existing Content Hash
# ============================================================

def _find_existing_hash(
    cursor,
    content_hash: str,
) -> dict | None:

    cursor.execute(
        sql.SQL(
            """
            SELECT
                id,
                source,
                source_id,
                title,
                content_hash
            FROM {}.{}
            WHERE content_hash = %s
            LIMIT 1
            """
        ).format(
            sql.Identifier(
                DB_SCHEMA
            ),
            sql.Identifier(
                DB_TABLE
            ),
        ),
        (
            content_hash,
        ),
    )

    row = (
        cursor.fetchone()
    )

    if row is None:

        return None

    return {
        "id": (
            row[0]
        ),

        "source": (
            row[1]
        ),

        "source_id": (
            row[2]
        ),

        "title": (
            row[3]
        ),

        "content_hash": (
            row[4]
        ),
    }


# ============================================================
# Build Dynamic UPSERT
# ============================================================

def _upsert_document(
    cursor,
    document: dict,
    columns: dict[str, dict],
) -> int:
    """
    실제 DB에 존재하는 column만 INSERT한다.

    UNIQUE(source, source_id)를 기준으로 UPSERT.

    created_at은 건드리지 않고,
    updated_at이 있다면 UPDATE 시 NOW() 처리한다.
    """

    insert_columns = [
        column_name
        for column_name in document.keys()
        if column_name in columns
    ]

    if not insert_columns:

        raise RuntimeError(
            "No compatible database columns."
        )

    adapted_values = [
        _adapt_value(
            document[
                column_name
            ],
            columns[
                column_name
            ],
        )
        for column_name
        in insert_columns
    ]

    # ========================================================
    # UPDATE columns
    # ========================================================

    update_columns = [
        column_name
        for column_name
        in insert_columns
        if column_name
        not in {
            "source",
            "source_id",
        }
    ]

    update_parts = [
        sql.SQL(
            "{} = EXCLUDED.{}"
        ).format(
            sql.Identifier(
                column_name
            ),
            sql.Identifier(
                column_name
            ),
        )
        for column_name
        in update_columns
    ]

    if (
        "updated_at"
        in columns
    ):

        update_parts.append(
            sql.SQL(
                "{} = NOW()"
            ).format(
                sql.Identifier(
                    "updated_at"
                )
            )
        )

    query = (
        sql.SQL(
            """
            INSERT INTO {}.{} ({})
            VALUES ({})
            ON CONFLICT (source, source_id)
            DO UPDATE SET {}
            RETURNING id
            """
        )
        .format(
            sql.Identifier(
                DB_SCHEMA
            ),

            sql.Identifier(
                DB_TABLE
            ),

            sql.SQL(
                ", "
            ).join(
                sql.Identifier(
                    column_name
                )
                for column_name
                in insert_columns
            ),

            sql.SQL(
                ", "
            ).join(
                sql.Placeholder()
                for _
                in insert_columns
            ),

            sql.SQL(
                ", "
            ).join(
                update_parts
            ),
        )
    )

    cursor.execute(
        query,
        adapted_values,
    )

    row = (
        cursor.fetchone()
    )

    if row is None:

        raise RuntimeError(
            "UPSERT returned no id."
        )

    return int(
        row[0]
    )


# ============================================================
# Database Summary
# ============================================================

def _get_database_summary(
    cursor,
) -> list[dict]:

    cursor.execute(
        sql.SQL(
            """
            SELECT
                source,
                topic_axis,
                COUNT(*) AS document_count
            FROM {}.{}
            GROUP BY
                source,
                topic_axis
            ORDER BY
                source,
                topic_axis
            """
        ).format(
            sql.Identifier(
                DB_SCHEMA
            ),
            sql.Identifier(
                DB_TABLE
            ),
        )
    )

    result = []

    for (
        source,
        topic_axis,
        count,
    ) in cursor.fetchall():

        result.append(
            {
                "source": (
                    source
                ),

                "topic_axis": (
                    topic_axis
                ),

                "count": int(
                    count
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
        "TEAM B - NASA NTRS DB Loader"
    )

    print("=" * 70)

    # ========================================================
    # 1. Local Documents
    # ========================================================

    cleaned_paths = (
        _find_cleaned_documents()
    )

    print(
        f"[LOCAL] Cleaned documents: "
        f"{len(cleaned_paths)}"
    )

    if not cleaned_paths:

        print(
            "[STOP] "
            "No cleaned NTRS documents found."
        )

        return

    try:

        documents = (
            _prepare_local_documents(
                cleaned_paths
            )
        )

    except Exception as exc:

        print()
        print(
            f"[STOP] "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return

    print(
        f"[LOCAL] Ready for DB: "
        f"{len(documents)}"
    )

    # 파일럿은 반드시 15편이어야 한다.
    if len(
        documents
    ) != 15:

        print()
        print(
            "[STOP] Expected exactly "
            "15 NTRS pilot documents."
        )

        print(
            f"[STOP] Found: "
            f"{len(documents)}"
        )

        return

    # ========================================================
    # Counters
    # ========================================================

    inserted = 0
    updated = 0
    duplicates = 0
    failed = 0

    duplicate_items = []
    loaded_items = []

    rows_before = 0
    rows_after = 0

    db_summary = []

    # ========================================================
    # 2. AWS RDS
    # ========================================================

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
            # One transaction:
            #
            # 15개 중 하나라도 DB write 자체가 실패하면
            # 전부 rollback.
            # =================================================

            with conn.transaction():

                with conn.cursor() as cursor:

                    # =========================================
                    # Schema
                    # =========================================

                    columns = (
                        _get_table_columns(
                            cursor
                        )
                    )

                    _validate_schema(
                        columns
                    )

                    print(
                        "[DB] core_documents "
                        "schema OK."
                    )

                    # =========================================
                    # Before
                    # =========================================

                    rows_before = (
                        _count_rows(
                            cursor
                        )
                    )

                    print(
                        f"[DB] Rows before: "
                        f"{rows_before}"
                    )

                    print()

                    # =========================================
                    # Load
                    # =========================================

                    for index, document in enumerate(
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

                        content_hash = (
                            document[
                                "content_hash"
                            ]
                        )

                        title = (
                            document[
                                "title"
                            ]
                        )

                        topic_axis = (
                            document[
                                "topic_axis"
                            ]
                        )

                        # =====================================
                        # A. Existing identity
                        # =====================================

                        existing_source = (
                            _find_existing_source(
                                cursor,
                                source,
                                source_id,
                            )
                        )

                        # =====================================
                        # B. Global content hash dedup
                        #
                        # 여기에서 기존 arXiv 15편과도 비교된다.
                        # =====================================

                        existing_hash = (
                            _find_existing_hash(
                                cursor,
                                content_hash,
                            )
                        )

                        # =====================================
                        # 동일 hash인데 다른 document identity
                        # =====================================

                        if (
                            existing_hash
                            is not None
                            and (
                                existing_hash[
                                    "source"
                                ]
                                != source
                                or existing_hash[
                                    "source_id"
                                ]
                                != source_id
                            )
                        ):

                            duplicates += 1

                            print(
                                f"[{index:02d}/"
                                f"{len(documents)}] "
                                f"DUPLICATE"
                            )

                            print(
                                f"    NTRS ID : "
                                f"{source_id}"
                            )

                            print(
                                f"    Title   : "
                                f"{title}"
                            )

                            print(
                                f"    Matches : "
                                f"{existing_hash['source']} / "
                                f"{existing_hash['source_id']}"
                            )

                            duplicate_items.append(
                                {
                                    "source": (
                                        source
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

                                    "content_hash": (
                                        content_hash
                                    ),

                                    "duplicate_of": (
                                        existing_hash
                                    ),
                                }
                            )

                            print()

                            continue

                        # =====================================
                        # INSERT / UPDATE
                        # =====================================

                        row_id = (
                            _upsert_document(
                                cursor,
                                document,
                                columns,
                            )
                        )

                        if (
                            existing_source
                            is None
                        ):

                            action = (
                                "INSERTED"
                            )

                            inserted += 1

                        else:

                            action = (
                                "UPDATED"
                            )

                            updated += 1

                        print(
                            f"[{index:02d}/"
                            f"{len(documents)}] "
                            f"{action}"
                        )

                        print(
                            f"    DB id   : "
                            f"{row_id}"
                        )

                        print(
                            f"    Axis    : "
                            f"{topic_axis}"
                        )

                        print(
                            f"    NTRS ID : "
                            f"{source_id}"
                        )

                        print(
                            f"    Title   : "
                            f"{title}"
                        )

                        print()

                        loaded_items.append(
                            {
                                "action": (
                                    action.lower()
                                ),

                                "db_id": (
                                    row_id
                                ),

                                "source": (
                                    source
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

                                "content_hash": (
                                    content_hash
                                ),
                            }
                        )

                    # =========================================
                    # After
                    # =========================================

                    rows_after = (
                        _count_rows(
                            cursor
                        )
                    )

                    db_summary = (
                        _get_database_summary(
                            cursor
                        )
                    )

            # transaction exits here -> COMMIT

    except Exception as exc:

        failed = len(
            documents
        )

        print()
        print("=" * 70)

        print(
            "DATABASE LOADING FAILED"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "The database transaction "
            "was rolled back."
        )

        report = {
            "source": "ntrs",

            "status": "failed",

            "local_documents": (
                len(
                    documents
                )
            ),

            "inserted": 0,

            "updated": 0,

            "duplicates": 0,

            "failed": (
                failed
            ),

            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        }

        _save_json(
            REPORT_FILE,
            report,
        )

        print(
            f"Report: "
            f"{REPORT_FILE}"
        )

        return

    # ========================================================
    # 3. Report
    # ========================================================

    report = {
        "source": "ntrs",

        "status": "success",

        "local_documents": (
            len(
                documents
            )
        ),

        "inserted": (
            inserted
        ),

        "updated": (
            updated
        ),

        "duplicates": (
            duplicates
        ),

        "failed": (
            failed
        ),

        "rows_before": (
            rows_before
        ),

        "rows_after": (
            rows_after
        ),

        "loaded_documents": (
            loaded_items
        ),

        "duplicate_documents": (
            duplicate_items
        ),

        "database_summary": (
            db_summary
        ),
    }

    _save_json(
        REPORT_FILE,
        report,
    )

    # ========================================================
    # Database Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "DATABASE SUMMARY"
    )

    print("=" * 70)

    for item in db_summary:

        print(
            f"{str(item['source']):8} | "
            f"{str(item['topic_axis']):22} | "
            f"{item['count']}"
        )

    # ========================================================
    # Final Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "NTRS DB LOADING COMPLETED"
    )

    print("=" * 70)

    print(
        f"Local documents : "
        f"{len(documents)}"
    )

    print(
        f"Inserted        : "
        f"{inserted}"
    )

    print(
        f"Updated         : "
        f"{updated}"
    )

    print(
        f"Duplicates      : "
        f"{duplicates}"
    )

    print(
        f"Failed          : "
        f"{failed}"
    )

    print(
        f"Rows before     : "
        f"{rows_before}"
    )

    print(
        f"Rows after      : "
        f"{rows_after}"
    )

    print(
        f"Report          : "
        f"{REPORT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()