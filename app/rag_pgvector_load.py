import json
from pathlib import Path

import numpy as np
import psycopg

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - Core-100 pgvector Loader
# Version: core100_pgvector_256_v1
# ============================================================
#
# INPUT
#
#   data/cache/core100_vectors_256.npy
#   data/cache/core100_chunk_vector_index.json
#
#
# OUTPUT
#
#   public.core_chunks.embedding
#
#
# Selected retrieval dimension:
#
#   256
#
#
# Current corpus:
#
#   documents = 100
#   chunks    = 3113
#
#
# SAFETY
#
# Before writing:
#
#   1. vector shape check
#   2. manifest count check
#   3. DB count check
#   4. chunk_id check
#   5. content_hash check
#   6. document_id check
#   7. chunk_index check
#
#
# This prevents loading stale vectors
# into a rebuilt chunk table.
# ============================================================


LOADER_VERSION = "core100_pgvector_256_v1"

EMBEDDING_VERSION = "core100_tfidf_svd_256_v1"

EXPECTED_CHUNKER_VERSION = "core100_section_v3"

VECTOR_DIMENSION = 256

EXPECTED_CHUNKS = 3113

EXPECTED_DOCUMENTS = 100

BATCH_SIZE = 100


# ============================================================
# Tables
# ============================================================

CHUNK_TABLE = "public.core_chunks"


# ============================================================
# Paths
# ============================================================

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
)


REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)


VECTOR_FILE = (
    CACHE_DIR
    / "core100_vectors_256.npy"
)


MANIFEST_FILE = (
    CACHE_DIR
    / "core100_chunk_vector_index.json"
)


REPORT_FILE = (
    REPORT_DIR
    / "core100_pgvector_load_report.json"
)


# ============================================================
# JSON
# ============================================================

def _load_json(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )

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
# Vector -> PostgreSQL vector literal
# ============================================================

def _vector_literal(
    vector: np.ndarray,
) -> str:

    return (
        "["
        + ",".join(
            f"{float(value):.9g}"
            for value in vector
        )
        + "]"
    )


# ============================================================
# Load Local Artifacts
# ============================================================

def _load_artifacts():

    if not VECTOR_FILE.exists():

        raise FileNotFoundError(
            f"Vector file missing: "
            f"{VECTOR_FILE}"
        )

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Manifest missing: "
            f"{MANIFEST_FILE}"
        )

    vectors = (
        np.load(
            VECTOR_FILE,
            allow_pickle=False,
        )
    )

    manifest_payload = (
        _load_json(
            MANIFEST_FILE
        )
    )

    items = (
        manifest_payload.get(
            "items"
        )
    )

    if not isinstance(
        items,
        list,
    ):

        raise RuntimeError(
            "Invalid vector manifest."
        )

    return (
        vectors,
        items,
    )


# ============================================================
# Local QA
# ============================================================

def _validate_local(
    vectors: np.ndarray,
    manifest: list[dict],
) -> dict:

    expected_shape = (
        EXPECTED_CHUNKS,
        VECTOR_DIMENSION,
    )

    if (
        vectors.shape
        != expected_shape
    ):

        raise RuntimeError(
            "Vector shape mismatch.\n"
            f"Expected: "
            f"{expected_shape}\n"
            f"Actual  : "
            f"{vectors.shape}"
        )

    if (
        len(
            manifest
        )
        != EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "Manifest count mismatch.\n"
            f"Expected: "
            f"{EXPECTED_CHUNKS}\n"
            f"Actual  : "
            f"{len(manifest)}"
        )

    if not (
        np.isfinite(
            vectors
        ).all()
    ):

        raise RuntimeError(
            "Vector array contains NaN or Inf."
        )

    norms = (
        np.linalg.norm(
            vectors,
            axis=1,
        )
    )

    zero_vectors = int(
        np.sum(
            norms
            < 1e-8
        )
    )

    if zero_vectors:

        raise RuntimeError(
            f"Zero vectors detected: "
            f"{zero_vectors}"
        )

    max_norm_error = float(
        np.max(
            np.abs(
                norms
                - 1.0
            )
        )
    )

    # vector_index alignment
    for (
        expected_index,
        item,
    ) in enumerate(
        manifest
    ):

        actual_index = int(
            item.get(
                "vector_index",
                -1,
            )
        )

        if (
            actual_index
            != expected_index
        ):

            raise RuntimeError(
                "vector_index mismatch.\n"
                f"Expected: "
                f"{expected_index}\n"
                f"Actual  : "
                f"{actual_index}"
            )

    return {
        "vector_shape": [
            int(
                vectors.shape[
                    0
                ]
            ),
            int(
                vectors.shape[
                    1
                ]
            ),
        ],

        "manifest_count": (
            len(
                manifest
            )
        ),

        "zero_vectors": (
            zero_vectors
        ),

        "max_l2_norm_error": (
            max_norm_error
        ),
    }


# ============================================================
# DB Schema Validation
# ============================================================

def _validate_schema(
    conn: psycopg.Connection,
) -> None:

    sql = """
    SELECT
        format_type(
            a.atttypid,
            a.atttypmod
        )

    FROM pg_attribute a

    JOIN pg_class c
      ON c.oid = a.attrelid

    JOIN pg_namespace n
      ON n.oid = c.relnamespace

    WHERE n.nspname = 'public'
      AND c.relname = 'core_chunks'
      AND a.attname = 'embedding'
      AND a.attnum > 0
      AND NOT a.attisdropped
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        row = (
            cur.fetchone()
        )

    if row is None:

        raise RuntimeError(
            "core_chunks.embedding does not exist.\n"
            "Run sql/005_pgvector_256.sql first."
        )

    column_type = str(
        row[
            0
        ]
    )

    expected_type = (
        f"vector("
        f"{VECTOR_DIMENSION}"
        f")"
    )

    if (
        column_type
        != expected_type
    ):

        raise RuntimeError(
            "Wrong embedding dimension.\n"
            f"Expected: "
            f"{expected_type}\n"
            f"Actual  : "
            f"{column_type}"
        )


# ============================================================
# Load DB Chunk Metadata
# ============================================================

def _load_db_chunks(
    conn: psycopg.Connection,
) -> dict[int, dict]:

    sql = f"""
    SELECT
        id,
        document_id,
        chunk_index,
        content_hash,
        chunker_version

    FROM {CHUNK_TABLE}

    WHERE chunker_version = %s

    ORDER BY
        document_id,
        chunk_index
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                EXPECTED_CHUNKER_VERSION,
            ),
        )

        rows = (
            cur.fetchall()
        )

    if (
        len(
            rows
        )
        != EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "DB chunk count mismatch.\n"
            f"Expected: "
            f"{EXPECTED_CHUNKS}\n"
            f"Actual  : "
            f"{len(rows)}"
        )

    result = {}

    for row in rows:

        (
            chunk_id,
            document_id,
            chunk_index,
            content_hash,
            chunker_version,
        ) = row

        result[
            int(
                chunk_id
            )
        ] = {
            "document_id": (
                int(
                    document_id
                )
            ),

            "chunk_index": (
                int(
                    chunk_index
                )
            ),

            "content_hash": (
                str(
                    content_hash
                )
            ),

            "chunker_version": (
                str(
                    chunker_version
                )
            ),
        }

    return result


# ============================================================
# Manifest vs DB Alignment
# ============================================================

def _validate_manifest_against_db(
    manifest: list[dict],
    db_chunks: dict[int, dict],
) -> None:

    mismatches = []

    document_ids = set()

    for item in manifest:

        chunk_id = int(
            item[
                "chunk_id"
            ]
        )

        document_id = int(
            item[
                "document_id"
            ]
        )

        chunk_index = int(
            item[
                "chunk_index"
            ]
        )

        content_hash = str(
            item[
                "content_hash"
            ]
        )

        document_ids.add(
            document_id
        )

        db_item = (
            db_chunks.get(
                chunk_id
            )
        )

        if db_item is None:

            mismatches.append(
                (
                    chunk_id,
                    "missing_chunk_id",
                )
            )

            continue

        if (
            db_item[
                "document_id"
            ]
            != document_id
        ):

            mismatches.append(
                (
                    chunk_id,
                    "document_id",
                )
            )

        if (
            db_item[
                "chunk_index"
            ]
            != chunk_index
        ):

            mismatches.append(
                (
                    chunk_id,
                    "chunk_index",
                )
            )

        if (
            db_item[
                "content_hash"
            ]
            != content_hash
        ):

            mismatches.append(
                (
                    chunk_id,
                    "content_hash",
                )
            )

    if (
        len(
            document_ids
        )
        != EXPECTED_DOCUMENTS
    ):

        raise RuntimeError(
            "Manifest document coverage mismatch.\n"
            f"Expected: "
            f"{EXPECTED_DOCUMENTS}\n"
            f"Actual  : "
            f"{len(document_ids)}"
        )

    if mismatches:

        raise RuntimeError(
            "Manifest does not match current "
            "core_chunks table.\n"
            f"Mismatch count: "
            f"{len(mismatches)}\n"
            f"Sample: "
            f"{mismatches[:10]}\n\n"
            "DO NOT load vectors.\n"
            "The chunk table may have been rebuilt "
            "after TF-IDF/SVD artifacts were created."
        )


# ============================================================
# Clear Previous Embeddings
# ============================================================

def _clear_embeddings(
    conn: psycopg.Connection,
) -> None:

    sql = f"""
    UPDATE {CHUNK_TABLE}
    SET
        embedding = NULL,
        embedding_version = NULL,
        embedded_at = NULL
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )


# ============================================================
# Write Vectors
# ============================================================

def _write_vectors(
    conn: psycopg.Connection,
    vectors: np.ndarray,
    manifest: list[dict],
) -> None:

    sql = f"""
    UPDATE {CHUNK_TABLE}
    SET
        embedding = %s::vector,
        embedding_version = %s,
        embedded_at = NOW()
    WHERE id = %s
    """

    total = (
        len(
            manifest
        )
    )

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start
            + BATCH_SIZE,
            total,
        )

        rows = []

        for vector_index in range(
            start,
            end,
        ):

            item = (
                manifest[
                    vector_index
                ]
            )

            vector = (
                vectors[
                    vector_index
                ]
            )

            rows.append(
                (
                    _vector_literal(
                        vector
                    ),

                    EMBEDDING_VERSION,

                    int(
                        item[
                            "chunk_id"
                        ]
                    ),
                )
            )

        with conn.cursor() as cur:

            cur.executemany(
                sql,
                rows,
            )

        print(
            f"[DB] Embedded "
            f"{end}/"
            f"{total}"
        )


# ============================================================
# DB Verification
# ============================================================

def _verify_database(
    conn: psycopg.Connection,
) -> dict:

    with conn.cursor() as cur:

        cur.execute(
            f"""
            SELECT
                COUNT(*) AS total,

                COUNT(*) FILTER (
                    WHERE embedding IS NOT NULL
                ) AS embedded,

                COUNT(*) FILTER (
                    WHERE embedding IS NULL
                ) AS missing,

                COUNT(*) FILTER (
                    WHERE embedding_version = %s
                ) AS correct_version

            FROM {CHUNK_TABLE}

            WHERE chunker_version = %s
            """,
            (
                EMBEDDING_VERSION,
                EXPECTED_CHUNKER_VERSION,
            ),
        )

        row = (
            cur.fetchone()
        )

    (
        total,
        embedded,
        missing,
        correct_version,
    ) = row

    result = {
        "total_chunks": (
            int(
                total
            )
        ),

        "embedded_chunks": (
            int(
                embedded
            )
        ),

        "missing_embeddings": (
            int(
                missing
            )
        ),

        "correct_version": (
            int(
                correct_version
            )
        ),
    }

    if (
        result[
            "total_chunks"
        ]
        != EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "Verification failed: "
            "unexpected chunk count."
        )

    if (
        result[
            "embedded_chunks"
        ]
        != EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "Verification failed: "
            "not all chunks have embeddings."
        )

    if (
        result[
            "missing_embeddings"
        ]
        != 0
    ):

        raise RuntimeError(
            "Verification failed: "
            "missing embeddings remain."
        )

    if (
        result[
            "correct_version"
        ]
        != EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "Verification failed: "
            "embedding version mismatch."
        )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - Core-100 "
        "pgvector Loader"
    )

    print("=" * 78)

    print(
        f"Loader version    : "
        f"{LOADER_VERSION}"
    )

    print(
        f"Embedding version : "
        f"{EMBEDDING_VERSION}"
    )

    print(
        f"Dimension         : "
        f"{VECTOR_DIMENSION}"
    )

    print(
        f"Expected chunks   : "
        f"{EXPECTED_CHUNKS}"
    )

    try:

        # ====================================================
        # Local
        # ====================================================

        print()
        print(
            "[LOCAL] Loading artifacts..."
        )

        (
            vectors,
            manifest,
        ) = (
            _load_artifacts()
        )

        local_qa = (
            _validate_local(
                vectors,
                manifest,
            )
        )

        print(
            f"[LOCAL] Vector shape : "
            f"{vectors.shape}"
        )

        print(
            f"[LOCAL] Manifest     : "
            f"{len(manifest)}"
        )

        print(
            f"[LOCAL] Zero vectors : "
            f"{local_qa['zero_vectors']}"
        )

        print(
            f"[LOCAL] Norm error   : "
            f"{local_qa['max_l2_norm_error']:.10f}"
        )

        # ====================================================
        # DB
        # ====================================================

        print()
        print(
            "[DB] Connecting to AWS RDS..."
        )

        with psycopg.connect(
            settings.dsn
        ) as conn:

            print(
                "[DB] Connected."
            )

            _validate_schema(
                conn
            )

            print(
                "[DB] vector(256) schema OK."
            )

            db_chunks = (
                _load_db_chunks(
                    conn
                )
            )

            _validate_manifest_against_db(
                manifest,
                db_chunks,
            )

            print(
                "[DB] Manifest ↔ DB alignment PASS."
            )

            # =================================================
            # Transaction
            # =================================================

            print()
            print(
                "[DB] Clearing previous embeddings "
                "inside transaction..."
            )

            _clear_embeddings(
                conn
            )

            print(
                "[DB] Writing vectors..."
            )

            _write_vectors(
                conn,
                vectors,
                manifest,
            )

            # =================================================
            # Verify before commit
            # =================================================

            verification = (
                _verify_database(
                    conn
                )
            )

            conn.commit()

        # ====================================================
        # Report
        # ====================================================

        report = {
            "loader_version": (
                LOADER_VERSION
            ),

            "embedding_version": (
                EMBEDDING_VERSION
            ),

            "dimension": (
                VECTOR_DIMENSION
            ),

            "local_qa": (
                local_qa
            ),

            "verification": (
                verification
            ),

            "vector_file": (
                str(
                    VECTOR_FILE
                )
            ),

            "manifest_file": (
                str(
                    MANIFEST_FILE
                )
            ),
        }

        _save_json(
            REPORT_FILE,
            report,
        )

        # ====================================================
        # Summary
        # ====================================================

        print()
        print("=" * 78)

        print(
            "PGVECTOR LOAD COMPLETED"
        )

        print("=" * 78)

        print(
            f"Total chunks       : "
            f"{verification['total_chunks']}"
        )

        print(
            f"Embedded chunks    : "
            f"{verification['embedded_chunks']}"
        )

        print(
            f"Missing embeddings : "
            f"{verification['missing_embeddings']}"
        )

        print(
            f"Correct version    : "
            f"{verification['correct_version']}"
        )

        print(
            f"Vector dimension   : "
            f"{VECTOR_DIMENSION}"
        )

        print()

        print(
            "[PASS] 3113 vectors stored "
            "in AWS RDS."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Run unseen-title Top-6 retrieval."
        )

        print("=" * 78)

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "PGVECTOR LOAD FAILED"
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

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()