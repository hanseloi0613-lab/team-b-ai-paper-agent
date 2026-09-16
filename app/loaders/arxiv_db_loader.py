import json
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from app.config import PROJECT_ROOT, settings


# ============================================================
# Database
# ============================================================

TABLE_SCHEMA = "public"
TABLE_NAME = "core_documents"


# ============================================================
# Paths
# ============================================================

RESOLVED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tmp"
    / "arxiv_resolved"
)

ARXIV_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "arxiv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "arxiv_db_loading_report.json"
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
    """
    JSON 파일 로드.
    """

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _save_json(
    path: Path,
    payload: dict,
) -> None:
    """
    JSON report 저장.
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
# arXiv ID Helpers
# ============================================================

def _strip_version(
    source_id: str,
) -> str:
    """
    arXiv version 제거.

    2401.11371v1
    ->
    2401.11371
    """

    import re

    return re.sub(
        r"v\d+$",
        "",
        source_id,
    )


# ============================================================
# Find Cleaned Documents
# ============================================================

def _find_cleaned_documents() -> list[Path]:
    """
    cleaner가 성공적으로 만든
    cleaned_document.json을 모두 찾는다.

    현재 파일럿에서는 15개가 나와야 한다.
    """

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            f"Resolved directory not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/cleaned_document.json"
        )
    )


# ============================================================
# Metadata Cache
# ============================================================

def _load_axis_metadata(
    topic_axis: str,
) -> list[dict]:
    """
    collector가 생성한 axis별 merged metadata 읽기.

    data/cache/arxiv/<axis>/_merged_candidates.json
    """

    path = (
        ARXIV_CACHE_ROOT
        / topic_axis
        / "_merged_candidates.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Metadata cache not found: {path}"
        )

    payload = _load_json(
        path
    )

    return payload.get(
        "documents",
        [],
    )


def _find_metadata(
    topic_axis: str,
    source_id: str,
) -> dict:
    """
    source_id에 해당하는 arXiv metadata 찾기.
    """

    documents = _load_axis_metadata(
        topic_axis
    )

    wanted_base = _strip_version(
        source_id
    )

    for document in documents:

        candidate_id = document.get(
            "source_id",
            "",
        )

        if candidate_id == source_id:
            return document

        if (
            _strip_version(
                candidate_id
            )
            == wanted_base
        ):
            return document

    raise LookupError(
        f"Metadata not found: "
        f"{topic_axis} / {source_id}"
    )


# ============================================================
# Resolution Metadata
# ============================================================

def _load_resolution(
    paper_dir: Path,
) -> dict:
    """
    Content Resolver 결과 읽기.
    """

    path = (
        paper_dir
        / "resolution.json"
    )

    if not path.exists():
        return {}

    return _load_json(
        path
    )


# ============================================================
# PostgreSQL Schema Inspection
# ============================================================

def _get_column_info(
    conn: psycopg.Connection,
) -> dict[str, dict]:
    """
    실제 AWS PostgreSQL의 core_documents
    컬럼 타입을 확인한다.

    authors/categories가 JSONB인지 TEXT[]인지
    코드가 추측하지 않고 DB에서 확인한다.
    """

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

        rows = cur.fetchall()

    if not rows:

        raise RuntimeError(
            f"Table not found: "
            f"{TABLE_SCHEMA}.{TABLE_NAME}\n"
            f"먼저 sql/002_schema.sql이 "
            f"AWS RDS에 실행됐는지 확인하세요."
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
            "data_type": data_type,
            "udt_name": udt_name,
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
    """
    loader가 필요로 하는 컬럼이
    DB에 실제로 존재하는지 확인.
    """

    existing_columns = set(
        column_info.keys()
    )

    missing = (
        REQUIRED_COLUMNS
        - existing_columns
    )

    if missing:

        missing_text = ", ".join(
            sorted(
                missing
            )
        )

        raise RuntimeError(
            "core_documents schema가 "
            "현재 loader와 맞지 않습니다.\n"
            f"Missing columns: "
            f"{missing_text}"
        )


# ============================================================
# PostgreSQL Type Adaptation
# ============================================================

def _adapt_collection(
    value: list | dict,
    column: dict,
) -> Any:
    """
    authors/categories/metadata 값을
    실제 PostgreSQL column type에 맞춘다.

    JSONB
        -> Jsonb

    ARRAY
        -> Python list

    TEXT/VARCHAR
        -> JSON string
    """

    data_type = (
        column.get(
            "data_type",
            "",
        )
        or ""
    ).lower()

    udt_name = (
        column.get(
            "udt_name",
            "",
        )
        or ""
    ).lower()

    # --------------------------------------------------------
    # JSON / JSONB
    # --------------------------------------------------------

    if data_type in {
        "json",
        "jsonb",
    }:

        return Jsonb(
            value
        )

    # --------------------------------------------------------
    # PostgreSQL ARRAY
    # --------------------------------------------------------

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
            str(value)
        ]

    # --------------------------------------------------------
    # TEXT / VARCHAR fallback
    # --------------------------------------------------------

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
    """
    YYYY-MM-DD -> date
    """

    if not value:
        return None

    try:

        return date.fromisoformat(
            value[:10]
        )

    except ValueError:

        return None


# ============================================================
# Build One DB Document
# ============================================================

def _build_document(
    cleaned_path: Path,
) -> dict:
    """
    한 논문에 필요한 모든 정보를 합친다.

    sources:

    1. merged arXiv metadata
    2. resolution.json
    3. raw_content.txt
    4. clean_content.txt
    5. cleaned_document.json
    """

    paper_dir = (
        cleaned_path.parent
    )

    topic_axis = (
        paper_dir
        .parent
        .name
    )

    cleaned = _load_json(
        cleaned_path
    )

    resolution = _load_resolution(
        paper_dir
    )

    source_id = (
        resolution.get(
            "source_id"
        )
        or paper_dir.name
    )

    metadata = _find_metadata(
        topic_axis,
        source_id,
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
            encoding="utf-8"
        )
    )

    clean_content = (
        clean_path.read_text(
            encoding="utf-8"
        )
    )

    quality = cleaned.get(
        "quality",
        {},
    )

    if not quality.get(
        "passed",
        False,
    ):

        raise ValueError(
            f"Quality check failed: "
            f"{source_id}"
        )

    content_hash = cleaned.get(
        "content_hash"
    )

    if not content_hash:

        raise ValueError(
            f"content_hash missing: "
            f"{source_id}"
        )

    # ========================================================
    # Extra provenance information
    # ========================================================

    original_metadata = metadata.get(
        "metadata",
        {},
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

    db_metadata = {
        **original_metadata,

        "matched_queries": metadata.get(
            "matched_queries",
            [],
        ),

        "source_base_id": metadata.get(
            "source_base_id"
        ),

        "content_resolution": {
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

        "quality": quality,

        "cleaning_stats": cleaned.get(
            "cleaning_stats",
            {},
        ),

        "pipeline": {
            "parser": (
                "arxiv_html_parser_v2"
            ),

            "normalization_version": (
                cleaned.get(
                    "normalization_version"
                )
            ),

            "loader": (
                "arxiv_db_loader_v1"
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

        "source_id": source_id,

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

        "authors": metadata.get(
            "authors",
            [],
        ),

        "categories": metadata.get(
            "categories",
            [],
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

        "doi": metadata.get(
            "doi"
        ),

        "url": metadata.get(
            "url"
        ),

        "pdf_url": metadata.get(
            "pdf_url"
        ),

        "topic_axis": topic_axis,

        "language": (
            metadata.get(
                "language"
            )
            or "en"
        ),

        "raw_content": raw_content,

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

        # parser + cleaner가 모두 성공했다는 의미
        "parse_status": "success",

        "metadata": db_metadata,
    }


# ============================================================
# Load All Local Documents
# ============================================================

def _load_local_documents() -> list[dict]:
    """
    현재 정제 완료된 논문을 전부 DB record로 구성한다.
    """

    cleaned_files = (
        _find_cleaned_documents()
    )

    print(
        f"[LOCAL] Cleaned documents: "
        f"{len(cleaned_files)}"
    )

    documents = []

    for cleaned_path in cleaned_files:

        document = _build_document(
            cleaned_path
        )

        documents.append(
            document
        )

    return documents


# ============================================================
# Existing Row Check
# ============================================================

def _find_existing_source(
    conn: psycopg.Connection,
    source: str,
    source_id: str,
) -> int | None:
    """
    같은 source + source_id가 이미 DB에 있는지 확인.
    """

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

        row = cur.fetchone()

    if row is None:
        return None

    return int(
        row[0]
    )


def _find_hash_owner(
    conn: psycopg.Connection,
    content_hash: str,
) -> tuple[int, str, str] | None:
    """
    DB에 같은 clean_content hash가 존재하는지 확인.
    """

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

        row = cur.fetchone()

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
    """
    source + source_id 기준 UPSERT.

    같은 arXiv 논문을 loader를 다시 실행해도
    row가 중복 생성되지 않는다.
    """

    authors = _adapt_collection(
        document[
            "authors"
        ],
        column_info[
            "authors"
        ],
    )

    categories = _adapt_collection(
        document[
            "categories"
        ],
        column_info[
            "categories"
        ],
    )

    metadata = _adapt_collection(
        document[
            "metadata"
        ],
        column_info[
            "metadata"
        ],
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

        row = cur.fetchone()

    if row is None:

        raise RuntimeError(
            f"UPSERT did not return id: "
            f"{document['source_id']}"
        )

    return int(
        row[0]
    )


# ============================================================
# DB Counts
# ============================================================

def _count_core_documents(
    conn: psycopg.Connection,
) -> int:
    """
    현재 DB core_documents 총 row 수.
    """

    sql = f"""
    SELECT COUNT(*)
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        row = cur.fetchone()

    return int(
        row[0]
    )


# ============================================================
# DB Summary
# ============================================================

def _print_database_summary(
    conn: psycopg.Connection,
) -> None:
    """
    INSERT 완료 후 axis별 개수 확인.
    """

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
    print("DATABASE SUMMARY")
    print("=" * 70)

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = cur.fetchall()

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
    """
    현재 파일럿 arXiv 15편을 AWS RDS에 적재한다.

    Pipeline:

    merged metadata
        +
    raw_content.txt
        +
    clean_content.txt
        +
    cleaned_document.json
        ↓
    schema validation
        ↓
    content_hash duplicate check
        ↓
    source + source_id UPSERT
        ↓
    AWS RDS core_documents
    """

    print()
    print("=" * 70)
    print("TEAM B - arXiv DB Loader")
    print("=" * 70)

    # ========================================================
    # Local Files
    # ========================================================

    documents = (
        _load_local_documents()
    )

    if not documents:

        print(
            "No cleaned documents found."
        )

        return

    print(
        f"[LOCAL] Ready for DB: "
        f"{len(documents)}"
    )

    # ========================================================
    # Connect AWS PostgreSQL
    # ========================================================

    print()
    print(
        "[DB] Connecting to AWS RDS..."
    )

    results = []

    try:

        with psycopg.connect(
            settings.dsn
        ) as conn:

            print(
                "[DB] Connected."
            )

            # =================================================
            # Schema Preflight
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

            before_count = (
                _count_core_documents(
                    conn
                )
            )

            print(
                f"[DB] Rows before: "
                f"{before_count}"
            )

            inserted_count = 0
            updated_count = 0
            duplicate_count = 0
            failed_count = 0

            # =================================================
            # Insert
            # =================================================

            for index, document in enumerate(
                documents,
                start=1,
            ):

                source = document[
                    "source"
                ]

                source_id = document[
                    "source_id"
                ]

                print()
                print(
                    "-" * 70
                )

                print(
                    f"[{index}/{len(documents)}] "
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
                    # Exact content duplicate belonging to
                    # another document
                    # ==========================================

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

                            duplicate_count += 1

                            print(
                                "[SKIP] Duplicate content hash."
                            )

                            print(
                                f"[SKIP] Existing DB id: "
                                f"{owner_id}"
                            )

                            print(
                                f"[SKIP] Existing document: "
                                f"{owner_source}/"
                                f"{owner_source_id}"
                            )

                            results.append(
                                {
                                    "status": (
                                        "duplicate"
                                    ),
                                    "source": source,
                                    "source_id": (
                                        source_id
                                    ),
                                    "duplicate_of_id": (
                                        owner_id
                                    ),
                                    "duplicate_of": (
                                        f"{owner_source}/"
                                        f"{owner_source_id}"
                                    ),
                                }
                            )

                            continue

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
                        f"[DB] id = {db_id}"
                    )

                    results.append(
                        {
                            "status": (
                                action.lower()
                            ),
                            "db_id": db_id,
                            "source": source,
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

                except Exception as exc:

                    failed_count += 1

                    print(
                        f"[ERROR] "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    # 한 문서 SQL 실패 후
                    # transaction 전체가 aborted 되는 것을 방지
                    conn.rollback()

                    # 이전 성공분은 아직 commit되지 않았으므로
                    # 단순 rollback하면 모두 날아간다.
                    #
                    # 따라서 오류가 발생하면 안전하게
                    # 전체 작업을 중단한다.
                    raise

            # =================================================
            # Commit
            # =================================================

            conn.commit()

            after_count = (
                _count_core_documents(
                    conn
                )
            )

            # =================================================
            # Verify
            # =================================================

            _print_database_summary(
                conn
            )

            print()
            print("=" * 70)
            print("DB LOADING COMPLETED")
            print("=" * 70)

            print(
                f"Local documents : "
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

            print(
                f"Rows before     : "
                f"{before_count}"
            )

            print(
                f"Rows after      : "
                f"{after_count}"
            )

            report_payload = {
                "local_documents": (
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
                "documents": results,
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

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            "DB LOADING FAILED"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "AWS RDS에 부분 데이터가 들어가는 것을 "
            "막기 위해 transaction을 취소했습니다."
        )

        print("=" * 70)

        raise


if __name__ == "__main__":
    main()