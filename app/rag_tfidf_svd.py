import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psycopg

from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - Core-100 TF-IDF + TruncatedSVD Experiment
# Version: core100_tfidf_svd_v1
# ============================================================
#
# INPUT
#
#   public.core_chunks
#
#   Expected:
#       chunker_version = core100_section_v3
#       documents       = 100
#       chunks          = 3113
#
#
# PIPELINE
#
#   retrieval_text
#        ↓
#   TF-IDF
#        ↓
#   Sparse matrix
#        ↓
#   TruncatedSVD
#        ├─ 128
#        ├─ 256
#        └─ 384
#        ↓
#   L2 normalization
#        ↓
#   quality comparison
#
#
# IMPORTANT
#
# - 아직 pgvector에 저장하지 않는다.
# - 128 / 256 / 384를 먼저 실제 corpus로 비교한다.
# - dimension 선택 후 다음 migration에서 vector(n)을 만든다.
#
#
# DIMENSION EVALUATION
#
# 1. Explained variance
#
# 2. TF-IDF 원공간의 Top-K 이웃을
#    SVD 공간이 얼마나 보존하는가
#
# 3. 저장 크기
#
# 4. smallest dimension within tolerance of best result
#
# ============================================================


PIPELINE_VERSION = "core100_tfidf_svd_v1"

EXPECTED_CHUNKER_VERSION = "core100_section_v3"

EXPECTED_DOCUMENTS = 100

EXPECTED_CHUNKS = 3113


# ============================================================
# TF-IDF Configuration
# ============================================================

TFIDF_MAX_FEATURES = 50_000

TFIDF_MIN_DF = 2

TFIDF_MAX_DF = 0.95

TFIDF_NGRAM_RANGE = (
    1,
    2,
)

TFIDF_SUBLINEAR_TF = True


# ============================================================
# SVD
# ============================================================

SVD_DIMENSIONS = (
    128,
    256,
    384,
)

RANDOM_STATE = 42


# ============================================================
# Evaluation
# ============================================================

NEIGHBOR_TOP_K = 10

NEIGHBOR_SAMPLE_SIZE = 300

# 최고 neighbor preservation과
# 0.02 이내라면 더 작은 차원을 우선 후보로 삼는다.
PROVISIONAL_TOLERANCE = 0.02


# ============================================================
# Tables
# ============================================================

CHUNK_TABLE = "public.core_chunks"

DOCUMENT_TABLE = "public.core_documents"


# ============================================================
# Artifact Paths
#
# 기존 data/cache, data/reports를 그대로 사용한다.
# 새 상위 폴더를 추가하지 않는다.
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


TFIDF_VECTORIZER_FILE = (
    CACHE_DIR
    / "core100_tfidf_vectorizer.joblib"
)


TFIDF_MATRIX_FILE = (
    CACHE_DIR
    / "core100_tfidf_matrix.npz"
)


CHUNK_INDEX_FILE = (
    CACHE_DIR
    / "core100_chunk_vector_index.json"
)


REPORT_FILE = (
    REPORT_DIR
    / "core100_tfidf_svd_report.json"
)


# ============================================================
# JSON
# ============================================================

def _save_json(
    path: Path,
    payload: Any,
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
# Helpers
# ============================================================

def _human_bytes(
    byte_count: int,
) -> str:

    value = float(
        byte_count
    )

    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    unit = units[0]

    for candidate in units:

        unit = candidate

        if value < 1024:
            break

        if candidate != units[-1]:

            value /= 1024

    return (
        f"{value:.2f} {unit}"
    )


def _file_size(
    path: Path,
) -> int:

    if not path.exists():

        return 0

    return int(
        path.stat().st_size
    )


# ============================================================
# DB Schema Validation
# ============================================================

def _validate_schema(
    conn: psycopg.Connection,
) -> None:

    required_chunk_columns = {
        "id",
        "document_id",
        "chunk_index",
        "retrieval_text",
        "content_hash",
        "chunker_version",
    }

    sql = """
    SELECT
        column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'core_chunks'
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = (
            cur.fetchall()
        )

    columns = {
        str(
            row[0]
        )
        for row in rows
    }

    if not columns:

        raise RuntimeError(
            "public.core_chunks does not exist."
        )

    missing = (
        required_chunk_columns
        - columns
    )

    if missing:

        raise RuntimeError(
            "core_chunks schema mismatch.\n"
            "Missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


# ============================================================
# Load Chunks
# ============================================================

def _load_chunks(
    conn: psycopg.Connection,
) -> list[dict]:

    sql = f"""
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.section_index,
        c.section_heading,
        c.retrieval_text,
        c.content_hash,
        c.chunker_version,

        d.source,
        d.source_id,
        d.title,
        d.topic_axis

    FROM {CHUNK_TABLE} c

    JOIN {DOCUMENT_TABLE} d
      ON d.id = c.document_id

    WHERE c.chunker_version = %s

    ORDER BY
        c.document_id,
        c.chunk_index
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

    chunks: list[dict] = []

    for row in rows:

        (
            chunk_id,
            document_id,
            chunk_index,
            section_index,
            section_heading,
            retrieval_text,
            content_hash,
            chunker_version,
            source,
            source_id,
            title,
            topic_axis,
        ) = row

        chunks.append(
            {
                "chunk_id": (
                    int(
                        chunk_id
                    )
                ),

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

                "section_index": (
                    int(
                        section_index
                    )
                ),

                "section_heading": (
                    str(
                        section_heading
                        or ""
                    )
                ),

                "retrieval_text": (
                    str(
                        retrieval_text
                        or ""
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

                "source": (
                    str(
                        source
                    )
                ),

                "source_id": (
                    str(
                        source_id
                    )
                ),

                "title": (
                    str(
                        title
                        or ""
                    )
                ),

                "topic_axis": (
                    str(
                        topic_axis
                        or ""
                    )
                ),
            }
        )

    return chunks


# ============================================================
# Corpus QA
# ============================================================

def _validate_chunks(
    chunks: list[dict],
) -> dict:

    if (
        len(
            chunks
        )
        != EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "Unexpected Core chunk count.\n"
            f"Expected: "
            f"{EXPECTED_CHUNKS}\n"
            f"Actual  : "
            f"{len(chunks)}"
        )

    document_ids = {
        chunk[
            "document_id"
        ]
        for chunk in chunks
    }

    if (
        len(
            document_ids
        )
        != EXPECTED_DOCUMENTS
    ):

        raise RuntimeError(
            "Unexpected document coverage.\n"
            f"Expected: "
            f"{EXPECTED_DOCUMENTS}\n"
            f"Actual  : "
            f"{len(document_ids)}"
        )

    empty = []

    hashes = set()

    duplicate_hashes = []

    wrong_version = []

    for chunk in chunks:

        if not (
            chunk[
                "retrieval_text"
            ].strip()
        ):

            empty.append(
                chunk[
                    "chunk_id"
                ]
            )

        digest = (
            chunk[
                "content_hash"
            ]
        )

        if digest in hashes:

            duplicate_hashes.append(
                digest
            )

        hashes.add(
            digest
        )

        if (
            chunk[
                "chunker_version"
            ]
            != EXPECTED_CHUNKER_VERSION
        ):

            wrong_version.append(
                chunk[
                    "chunk_id"
                ]
            )

    if empty:

        raise RuntimeError(
            "Empty retrieval_text detected.\n"
            f"Count: {len(empty)}"
        )

    if duplicate_hashes:

        raise RuntimeError(
            "Duplicate content hashes detected "
            "before TF-IDF.\n"
            f"Count: "
            f"{len(duplicate_hashes)}"
        )

    if wrong_version:

        raise RuntimeError(
            "Unexpected chunker version detected."
        )

    return {
        "chunks": (
            len(
                chunks
            )
        ),

        "documents": (
            len(
                document_ids
            )
        ),

        "empty_retrieval_text": (
            len(
                empty
            )
        ),

        "duplicate_hashes": (
            len(
                duplicate_hashes
            )
        ),
    }


# ============================================================
# Chunk Index Manifest
# ============================================================

def _build_chunk_index(
    chunks: list[dict],
) -> list[dict]:

    result = []

    for (
        vector_index,
        chunk,
    ) in enumerate(
        chunks
    ):

        result.append(
            {
                "vector_index": (
                    vector_index
                ),

                "chunk_id": (
                    chunk[
                        "chunk_id"
                    ]
                ),

                "document_id": (
                    chunk[
                        "document_id"
                    ]
                ),

                "chunk_index": (
                    chunk[
                        "chunk_index"
                    ]
                ),

                "section_index": (
                    chunk[
                        "section_index"
                    ]
                ),

                "section_heading": (
                    chunk[
                        "section_heading"
                    ]
                ),

                "source": (
                    chunk[
                        "source"
                    ]
                ),

                "source_id": (
                    chunk[
                        "source_id"
                    ]
                ),

                "topic_axis": (
                    chunk[
                        "topic_axis"
                    ]
                ),

                "title": (
                    chunk[
                        "title"
                    ]
                ),

                "content_hash": (
                    chunk[
                        "content_hash"
                    ]
                ),
            }
        )

    return result


# ============================================================
# TF-IDF
# ============================================================

def _fit_tfidf(
    texts: list[str],
) -> tuple[
    TfidfVectorizer,
    sparse.csr_matrix,
]:

    vectorizer = (
        TfidfVectorizer(
            lowercase=True,

            strip_accents="unicode",

            analyzer="word",

            ngram_range=(
                TFIDF_NGRAM_RANGE
            ),

            min_df=(
                TFIDF_MIN_DF
            ),

            max_df=(
                TFIDF_MAX_DF
            ),

            max_features=(
                TFIDF_MAX_FEATURES
            ),

            sublinear_tf=(
                TFIDF_SUBLINEAR_TF
            ),

            norm="l2",

            dtype=np.float32,
        )
    )

    matrix = (
        vectorizer.fit_transform(
            texts
        )
    )

    matrix = (
        matrix.tocsr()
    )

    return (
        vectorizer,
        matrix,
    )


# ============================================================
# TF-IDF Statistics
# ============================================================

def _tfidf_stats(
    matrix: sparse.csr_matrix,
    vectorizer: TfidfVectorizer,
) -> dict:

    rows = int(
        matrix.shape[0]
    )

    columns = int(
        matrix.shape[1]
    )

    nnz = int(
        matrix.nnz
    )

    total_cells = (
        rows
        * columns
    )

    density = (
        float(
            nnz
        )
        / float(
            total_cells
        )
        if total_cells
        else 0.0
    )

    vocabulary_size = (
        len(
            vectorizer.vocabulary_
        )
    )

    return {
        "shape": [
            rows,
            columns,
        ],

        "chunks": (
            rows
        ),

        "features": (
            columns
        ),

        "vocabulary_size": (
            vocabulary_size
        ),

        "non_zero_values": (
            nnz
        ),

        "density": (
            round(
                density,
                8,
            )
        ),
    }


# ============================================================
# Top-K Helpers
# ============================================================

def _top_k_from_scores(
    scores: np.ndarray,
    query_index: int,
    top_k: int,
) -> np.ndarray:

    scores = (
        scores.copy()
    )

    # 자기 자신 제외
    scores[
        query_index
    ] = -np.inf

    # 후보가 충분히 많으므로 argpartition 사용
    candidate_indices = (
        np.argpartition(
            scores,
            -top_k,
        )[
            -top_k:
        ]
    )

    candidate_scores = (
        scores[
            candidate_indices
        ]
    )

    order = (
        np.argsort(
            candidate_scores
        )[
            ::-1
        ]
    )

    return (
        candidate_indices[
            order
        ]
    )


# ============================================================
# Reference TF-IDF Neighbors
# ============================================================

def _build_reference_neighbors(
    tfidf_matrix: sparse.csr_matrix,
    sample_indices: np.ndarray,
) -> dict[int, set[int]]:

    print()
    print(
        "[EVAL] Building TF-IDF "
        "reference neighbors..."
    )

    # TF-IDF matrix가 이미 L2 norm이므로
    # dot product == cosine similarity
    similarities = (
        tfidf_matrix[
            sample_indices
        ]
        @ tfidf_matrix.T
    )

    similarities = (
        similarities
        .toarray()
    )

    reference: dict[
        int,
        set[int],
    ] = {}

    for (
        row_index,
        corpus_index,
    ) in enumerate(
        sample_indices
    ):

        top_indices = (
            _top_k_from_scores(
                similarities[
                    row_index
                ],
                int(
                    corpus_index
                ),
                NEIGHBOR_TOP_K,
            )
        )

        reference[
            int(
                corpus_index
            )
        ] = {
            int(x)
            for x in top_indices
        }

    return reference


# ============================================================
# SVD Neighbor Preservation
# ============================================================

def _neighbor_preservation(
    vectors: np.ndarray,
    sample_indices: np.ndarray,
    reference_neighbors: dict[
        int,
        set[int],
    ],
) -> dict:

    # vectors are L2 normalized.
    similarities = (
        vectors[
            sample_indices
        ]
        @ vectors.T
    )

    overlap_scores = []

    exact_top1 = 0

    for (
        row_index,
        corpus_index,
    ) in enumerate(
        sample_indices
    ):

        corpus_index = int(
            corpus_index
        )

        predicted = (
            _top_k_from_scores(
                similarities[
                    row_index
                ],
                corpus_index,
                NEIGHBOR_TOP_K,
            )
        )

        predicted_set = {
            int(x)
            for x in predicted
        }

        reference_set = (
            reference_neighbors[
                corpus_index
            ]
        )

        overlap = (
            len(
                predicted_set
                & reference_set
            )
            / NEIGHBOR_TOP_K
        )

        overlap_scores.append(
            overlap
        )

        reference_first = next(
            iter(
                reference_set
            )
        )

        # 아래 top1은 보조 지표.
        # reference_set 자체는 unordered이므로
        # 핵심 선택지표는 overlap.
        if (
            int(
                predicted[0]
            )
            == reference_first
        ):

            exact_top1 += 1

    return {
        "top_k": (
            NEIGHBOR_TOP_K
        ),

        "sample_size": (
            len(
                sample_indices
            )
        ),

        "mean_topk_overlap": (
            round(
                float(
                    np.mean(
                        overlap_scores
                    )
                ),
                6,
            )
        ),

        "median_topk_overlap": (
            round(
                float(
                    np.median(
                        overlap_scores
                    )
                ),
                6,
            )
        ),

        "min_topk_overlap": (
            round(
                float(
                    np.min(
                        overlap_scores
                    )
                ),
                6,
            )
        ),

        "max_topk_overlap": (
            round(
                float(
                    np.max(
                        overlap_scores
                    )
                ),
                6,
            )
        ),
    }


# ============================================================
# Fit One SVD
# ============================================================

def _fit_one_svd(
    tfidf_matrix: sparse.csr_matrix,
    dimension: int,
    sample_indices: np.ndarray,
    reference_neighbors: dict[
        int,
        set[int],
    ],
) -> dict:

    print()
    print("-" * 78)

    print(
        f"[SVD] Fitting dimension "
        f"{dimension}..."
    )

    if (
        dimension
        >= tfidf_matrix.shape[1]
    ):

        raise RuntimeError(
            f"SVD dimension {dimension} "
            "must be smaller than "
            f"feature count "
            f"{tfidf_matrix.shape[1]}."
        )

    if (
        dimension
        >= tfidf_matrix.shape[0]
    ):

        raise RuntimeError(
            f"SVD dimension {dimension} "
            "must be smaller than "
            f"sample count "
            f"{tfidf_matrix.shape[0]}."
        )

    svd = (
        TruncatedSVD(
            n_components=(
                dimension
            ),

            algorithm="randomized",

            n_iter=7,

            random_state=(
                RANDOM_STATE
            ),
        )
    )

    reduced = (
        svd.fit_transform(
            tfidf_matrix
        )
    )

    reduced = (
        reduced.astype(
            np.float32,
            copy=False,
        )
    )

    # pgvector cosine search를 위해
    # 이후에도 이 normalized vector를 사용한다.
    vectors = (
        normalize(
            reduced,
            norm="l2",
            axis=1,
            copy=False,
        )
    )

    vectors = (
        np.asarray(
            vectors,
            dtype=np.float32,
        )
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    if not (
        np.isfinite(
            vectors
        ).all()
    ):

        raise RuntimeError(
            f"SVD {dimension}: "
            "NaN/Inf detected."
        )

    row_norms = (
        np.linalg.norm(
            vectors,
            axis=1,
        )
    )

    zero_vectors = int(
        np.sum(
            row_norms
            < 1e-8
        )
    )

    if zero_vectors:

        raise RuntimeError(
            f"SVD {dimension}: "
            f"{zero_vectors} zero vector(s)."
        )

    max_norm_error = float(
        np.max(
            np.abs(
                row_norms
                - 1.0
            )
        )
    )

    explained_variance = float(
        np.sum(
            svd
            .explained_variance_ratio_
        )
    )

    neighbor_metrics = (
        _neighbor_preservation(
            vectors,
            sample_indices,
            reference_neighbors,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    svd_file = (
        CACHE_DIR
        / f"core100_svd_{dimension}.joblib"
    )

    vector_file = (
        CACHE_DIR
        / f"core100_vectors_{dimension}.npy"
    )

    joblib.dump(
        svd,
        svd_file,
    )

    np.save(
        vector_file,
        vectors,
        allow_pickle=False,
    )

    svd_bytes = (
        _file_size(
            svd_file
        )
    )

    vector_bytes = (
        _file_size(
            vector_file
        )
    )

    print(
        f"[SVD] Explained variance : "
        f"{explained_variance:.6f}"
    )

    print(
        f"[SVD] Neighbor overlap   : "
        f"{neighbor_metrics['mean_topk_overlap']:.6f}"
    )

    print(
        f"[SVD] Zero vectors       : "
        f"{zero_vectors}"
    )

    print(
        f"[SVD] Max norm error     : "
        f"{max_norm_error:.8f}"
    )

    print(
        f"[SVD] Vector shape       : "
        f"{vectors.shape}"
    )

    print(
        f"[SVD] Vector file        : "
        f"{_human_bytes(vector_bytes)}"
    )

    return {
        "dimension": (
            dimension
        ),

        "explained_variance_ratio_sum": (
            round(
                explained_variance,
                8,
            )
        ),

        "neighbor_preservation": (
            neighbor_metrics
        ),

        "zero_vectors": (
            zero_vectors
        ),

        "max_l2_norm_error": (
            round(
                max_norm_error,
                10,
            )
        ),

        "vector_shape": [
            int(
                vectors.shape[0]
            ),
            int(
                vectors.shape[1]
            ),
        ],

        "svd_model_file": (
            str(
                svd_file
            )
        ),

        "vector_file": (
            str(
                vector_file
            )
        ),

        "svd_model_bytes": (
            svd_bytes
        ),

        "vector_bytes": (
            vector_bytes
        ),

        "vector_size_human": (
            _human_bytes(
                vector_bytes
            )
        ),
    }


# ============================================================
# Provisional Dimension
# ============================================================

def _select_provisional_dimension(
    results: list[dict],
) -> dict:

    if not results:

        raise RuntimeError(
            "No SVD results."
        )

    best_overlap = max(
        result[
            "neighbor_preservation"
        ][
            "mean_topk_overlap"
        ]
        for result in results
    )

    threshold = (
        best_overlap
        - PROVISIONAL_TOLERANCE
    )

    acceptable = [
        result
        for result in results
        if (
            result[
                "neighbor_preservation"
            ][
                "mean_topk_overlap"
            ]
            >= threshold
        )
    ]

    acceptable.sort(
        key=lambda x: (
            x[
                "dimension"
            ]
        )
    )

    selected = (
        acceptable[0]
    )

    return {
        "dimension": (
            selected[
                "dimension"
            ]
        ),

        "reason": (
            "Smallest tested dimension within "
            f"{PROVISIONAL_TOLERANCE:.2f} "
            "absolute mean Top-K neighbor overlap "
            "of the best tested dimension."
        ),

        "best_neighbor_overlap": (
            best_overlap
        ),

        "selection_threshold": (
            round(
                threshold,
                6,
            )
        ),

        "IMPORTANT": (
            "This is provisional only. "
            "Final dimension must be confirmed "
            "with actual retrieval evaluation "
            "before pgvector migration."
        ),
    }


# ============================================================
# Console Summary
# ============================================================

def _print_summary(
    corpus_qa: dict,
    tfidf_stats: dict,
    svd_results: list[dict],
    provisional: dict,
) -> None:

    print()
    print("=" * 78)

    print(
        "CORE-100 TF-IDF / SVD EXPERIMENT COMPLETED"
    )

    print("=" * 78)

    print(
        f"Pipeline version : "
        f"{PIPELINE_VERSION}"
    )

    print(
        f"Chunks           : "
        f"{corpus_qa['chunks']}"
    )

    print(
        f"Documents        : "
        f"{corpus_qa['documents']}"
    )

    print(
        f"Vocabulary       : "
        f"{tfidf_stats['vocabulary_size']}"
    )

    print(
        f"TF-IDF shape     : "
        f"{tfidf_stats['shape']}"
    )

    print(
        f"TF-IDF density   : "
        f"{tfidf_stats['density']}"
    )

    print()
    print(
        "SVD comparison:"
    )

    print(
        "-" * 78
    )

    print(
        f"{'DIM':>6} | "
        f"{'EXPL.VAR':>10} | "
        f"{'TOP10 OVERLAP':>13} | "
        f"{'VECTOR SIZE':>12}"
    )

    print(
        "-" * 78
    )

    for result in svd_results:

        print(
            f"{result['dimension']:>6} | "
            f"{result['explained_variance_ratio_sum']:>10.6f} | "
            f"{result['neighbor_preservation']['mean_topk_overlap']:>13.6f} | "
            f"{result['vector_size_human']:>12}"
        )

    print(
        "-" * 78
    )

    print()

    print(
        "Provisional dimension:"
    )

    print(
        f"  {provisional['dimension']}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  This is NOT the final pgvector dimension."
    )

    print(
        "  Final selection happens after retrieval evaluation."
    )

    print()

    print(
        "Artifacts:"
    )

    print(
        f"  TF-IDF vectorizer : "
        f"{TFIDF_VECTORIZER_FILE}"
    )

    print(
        f"  TF-IDF matrix     : "
        f"{TFIDF_MATRIX_FILE}"
    )

    print(
        f"  Chunk index       : "
        f"{CHUNK_INDEX_FILE}"
    )

    print(
        f"  Report            : "
        f"{REPORT_FILE}"
    )

    print("=" * 78)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - Core-100 "
        "TF-IDF + TruncatedSVD Experiment"
    )

    print("=" * 78)

    print(
        f"Pipeline version : "
        f"{PIPELINE_VERSION}"
    )

    print(
        f"Chunker required : "
        f"{EXPECTED_CHUNKER_VERSION}"
    )

    print(
        f"Expected docs    : "
        f"{EXPECTED_DOCUMENTS}"
    )

    print(
        f"Expected chunks  : "
        f"{EXPECTED_CHUNKS}"
    )

    print(
        f"SVD dimensions   : "
        f"{SVD_DIMENSIONS}"
    )

    print()

    print(
        "[DB] Connecting to AWS RDS..."
    )

    try:

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # Load Corpus
        # ====================================================

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
                "[DB] Schema OK."
            )

            chunks = (
                _load_chunks(
                    conn
                )
            )

        # DB connection no longer needed during ML fitting.

        print(
            f"[DB] Loaded chunks: "
            f"{len(chunks)}"
        )

        # ====================================================
        # Corpus QA
        # ====================================================

        corpus_qa = (
            _validate_chunks(
                chunks
            )
        )

        print()
        print(
            "[QA] Core-100 chunk corpus PASS."
        )

        print(
            f"[QA] Documents : "
            f"{corpus_qa['documents']}"
        )

        print(
            f"[QA] Chunks    : "
            f"{corpus_qa['chunks']}"
        )

        print(
            f"[QA] Empty     : "
            f"{corpus_qa['empty_retrieval_text']}"
        )

        print(
            f"[QA] Duplicates: "
            f"{corpus_qa['duplicate_hashes']}"
        )

        # ====================================================
        # Index Manifest
        # ====================================================

        chunk_index = (
            _build_chunk_index(
                chunks
            )
        )

        _save_json(
            CHUNK_INDEX_FILE,
            {
                "pipeline_version": (
                    PIPELINE_VERSION
                ),

                "chunker_version": (
                    EXPECTED_CHUNKER_VERSION
                ),

                "count": (
                    len(
                        chunk_index
                    )
                ),

                "items": (
                    chunk_index
                ),
            },
        )

        # ====================================================
        # TF-IDF
        # ====================================================

        texts = [
            chunk[
                "retrieval_text"
            ]
            for chunk in chunks
        ]

        print()
        print(
            "[TF-IDF] Fitting vocabulary..."
        )

        (
            vectorizer,
            tfidf_matrix,
        ) = (
            _fit_tfidf(
                texts
            )
        )

        tfidf_stats = (
            _tfidf_stats(
                tfidf_matrix,
                vectorizer,
            )
        )

        print(
            f"[TF-IDF] Matrix shape : "
            f"{tfidf_matrix.shape}"
        )

        print(
            f"[TF-IDF] Vocabulary   : "
            f"{tfidf_stats['vocabulary_size']}"
        )

        print(
            f"[TF-IDF] NNZ          : "
            f"{tfidf_stats['non_zero_values']}"
        )

        print(
            f"[TF-IDF] Density      : "
            f"{tfidf_stats['density']}"
        )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        max_requested_dimension = max(
            SVD_DIMENSIONS
        )

        if (
            tfidf_matrix.shape[1]
            <= max_requested_dimension
        ):

            raise RuntimeError(
                "TF-IDF vocabulary too small "
                "for requested SVD experiment.\n"
                f"Features : "
                f"{tfidf_matrix.shape[1]}\n"
                f"Required : "
                f"> {max_requested_dimension}"
            )

        # ====================================================
        # Save TF-IDF
        # ====================================================

        joblib.dump(
            vectorizer,
            TFIDF_VECTORIZER_FILE,
        )

        sparse.save_npz(
            TFIDF_MATRIX_FILE,
            tfidf_matrix,
            compressed=True,
        )

        print()
        print(
            "[TF-IDF] Saved."
        )

        print(
            f"[TF-IDF] Vectorizer : "
            f"{TFIDF_VECTORIZER_FILE}"
        )

        print(
            f"[TF-IDF] Matrix     : "
            f"{TFIDF_MATRIX_FILE}"
        )

        # ====================================================
        # Evaluation Sample
        # ====================================================

        rng = (
            np.random.default_rng(
                RANDOM_STATE
            )
        )

        sample_size = min(
            NEIGHBOR_SAMPLE_SIZE,
            tfidf_matrix.shape[0],
        )

        sample_indices = (
            rng.choice(
                tfidf_matrix.shape[0],
                size=sample_size,
                replace=False,
            )
        )

        sample_indices = (
            np.sort(
                sample_indices
            )
        )

        reference_neighbors = (
            _build_reference_neighbors(
                tfidf_matrix,
                sample_indices,
            )
        )

        # ====================================================
        # Fit SVD Experiments
        # ====================================================

        svd_results: list[dict] = []

        for dimension in SVD_DIMENSIONS:

            result = (
                _fit_one_svd(
                    tfidf_matrix,
                    dimension,
                    sample_indices,
                    reference_neighbors,
                )
            )

            svd_results.append(
                result
            )

        # ====================================================
        # Provisional Selection
        # ====================================================

        provisional = (
            _select_provisional_dimension(
                svd_results
            )
        )

        # ====================================================
        # Report
        # ====================================================

        report = {
            "pipeline_version": (
                PIPELINE_VERSION
            ),

            "chunker_version": (
                EXPECTED_CHUNKER_VERSION
            ),

            "corpus": (
                corpus_qa
            ),

            "tfidf": {
                "configuration": {
                    "max_features": (
                        TFIDF_MAX_FEATURES
                    ),

                    "min_df": (
                        TFIDF_MIN_DF
                    ),

                    "max_df": (
                        TFIDF_MAX_DF
                    ),

                    "ngram_range": list(
                        TFIDF_NGRAM_RANGE
                    ),

                    "sublinear_tf": (
                        TFIDF_SUBLINEAR_TF
                    ),

                    "norm": (
                        "l2"
                    ),
                },

                "statistics": (
                    tfidf_stats
                ),

                "vectorizer_file": (
                    str(
                        TFIDF_VECTORIZER_FILE
                    )
                ),

                "matrix_file": (
                    str(
                        TFIDF_MATRIX_FILE
                    )
                ),
            },

            "svd": {
                "dimensions_tested": list(
                    SVD_DIMENSIONS
                ),

                "random_state": (
                    RANDOM_STATE
                ),

                "results": (
                    svd_results
                ),
            },

            "neighbor_evaluation": {
                "sample_size": (
                    sample_size
                ),

                "top_k": (
                    NEIGHBOR_TOP_K
                ),

                "reference_space": (
                    "L2-normalized TF-IDF cosine"
                ),
            },

            "provisional_dimension": (
                provisional
            ),

            "chunk_index_file": (
                str(
                    CHUNK_INDEX_FILE
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

        _print_summary(
            corpus_qa,
            tfidf_stats,
            svd_results,
            provisional,
        )

        print()
        print(
            "[PASS] TF-IDF/SVD experiment completed."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Send this console result before "
            "creating the pgvector column."
        )

        print(
            "We will select the actual vector dimension "
            "from the measured results."
        )

        print("=" * 78)

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "TF-IDF / SVD EXPERIMENT FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "No pgvector schema change was made."
        )

        print(
            "Do NOT create the vector column yet."
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()