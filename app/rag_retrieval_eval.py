import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scipy import sparse
from sklearn.preprocessing import normalize

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - Core-100 Retrieval Evaluation
# Version: core100_retrieval_eval_v2
# ============================================================
#
# PURPOSE
#
# Compare:
#
#   TF-IDF
#   SVD-128
#   SVD-256
#   SVD-384
#
#
# Current frozen corpus:
#
#   documents : 100
#   chunks    : 3113
#
#
# Query:
#
#   core document title
#
#
# IMPORTANT
#
# Query document's own chunks are EXCLUDED.
#
# Otherwise:
#
#   paper title
#       ↓
#   its own chunks
#
# would trivially rank highly and evaluation would be fake.
#
#
# Retrieval:
#
#   title query
#       ↓
#   TF-IDF
#       ↓
#   optional SVD projection
#       ↓
#   cosine similarity against chunks
#       ↓
#   chunk scores
#       ↓
#   document-wise MAX score
#       ↓
#   Top-K documents
#
#
# Why document-wise MAX?
#
# Long papers may have 80 chunks.
# Short papers may have only 5 chunks.
#
# If raw chunk Top-K is used,
# long papers get more chances to occupy Top-K.
#
# Therefore:
#
#   one document
#   =
#   best matching chunk score
#
#
# Automatic relevance proxy:
#
#   same topic_axis
#
#   rover_autonomy
#   onboard_ai
#   satellite_autonomy
#
#
# NOTE:
#
# This proxy is NOT final scientific relevance annotation.
#
# It is used for:
#
#   - dimension comparison
#   - retrieval sanity check
#   - RAG engineering baseline
#
#
# Later:
#
# manually curated queries + qrels
#
# for:
#
#   Recall@K
#   nDCG@K
#   MRR
#   evidence coverage
#
# ============================================================


EVALUATOR_VERSION = "core100_retrieval_eval_v2"

EXPECTED_CHUNKS = 3113

EXPECTED_DOCUMENTS = 100

TOP_K = 6


# ============================================================
# SVD candidates
# ============================================================

DIMENSIONS = (
    128,
    256,
    384,
)


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
    / "core100_retrieval_eval_report.json"
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
            encoding="utf-8",
        )
    )


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
# Artifact Preflight
# ============================================================

def _validate_artifacts() -> None:

    required = [
        TFIDF_VECTORIZER_FILE,
        TFIDF_MATRIX_FILE,
        CHUNK_INDEX_FILE,
    ]

    for dimension in DIMENSIONS:

        required.append(
            CACHE_DIR
            / f"core100_svd_{dimension}.joblib"
        )

        required.append(
            CACHE_DIR
            / f"core100_vectors_{dimension}.npy"
        )

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        raise RuntimeError(
            "Required artifact(s) missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


# ============================================================
# Load Chunk Manifest
# ============================================================

def _load_manifest() -> list[dict]:

    payload = (
        _load_json(
            CHUNK_INDEX_FILE
        )
    )

    items = (
        payload.get(
            "items"
        )
    )

    if not isinstance(
        items,
        list,
    ):

        raise RuntimeError(
            "Invalid chunk index manifest."
        )

    if len(
        items
    ) != EXPECTED_CHUNKS:

        raise RuntimeError(
            "Chunk manifest count mismatch.\n"
            f"Expected: {EXPECTED_CHUNKS}\n"
            f"Actual  : {len(items)}"
        )

    # ========================================================
    # vector_index must be continuous
    # ========================================================

    for (
        expected_index,
        item,
    ) in enumerate(
        items
    ):

        actual_index = int(
            item.get(
                "vector_index",
                -1,
            )
        )

        if actual_index != expected_index:

            raise RuntimeError(
                "Broken vector_index sequence.\n"
                f"Expected: {expected_index}\n"
                f"Actual  : {actual_index}"
            )

    return items


# ============================================================
# Document Index
# ============================================================

def _build_document_index(
    manifest: list[dict],
) -> dict[int, dict]:

    documents: dict[
        int,
        dict,
    ] = {}

    for item in manifest:

        document_id = int(
            item[
                "document_id"
            ]
        )

        metadata = {
            "document_id": (
                document_id
            ),

            "source": (
                str(
                    item[
                        "source"
                    ]
                )
            ),

            "source_id": (
                str(
                    item[
                        "source_id"
                    ]
                )
            ),

            "title": (
                str(
                    item[
                        "title"
                    ]
                )
            ),

            "topic_axis": (
                str(
                    item[
                        "topic_axis"
                    ]
                )
            ),
        }

        existing = (
            documents.get(
                document_id
            )
        )

        if existing is None:

            documents[
                document_id
            ] = metadata

            continue

        for key in (
            "source",
            "source_id",
            "title",
            "topic_axis",
        ):

            if (
                existing[
                    key
                ]
                != metadata[
                    key
                ]
            ):

                raise RuntimeError(
                    "Inconsistent document metadata.\n"
                    f"document_id={document_id}\n"
                    f"field={key}"
                )

    if len(
        documents
    ) != EXPECTED_DOCUMENTS:

        raise RuntimeError(
            "Document count mismatch.\n"
            f"Expected: {EXPECTED_DOCUMENTS}\n"
            f"Actual  : {len(documents)}"
        )

    return documents


# ============================================================
# Corpus Distribution
# ============================================================

def _axis_document_counts(
    documents: dict[int, dict],
) -> dict[str, int]:

    result: dict[
        str,
        int,
    ] = {}

    for document in documents.values():

        axis = (
            document[
                "topic_axis"
            ]
        )

        result[
            axis
        ] = (
            result.get(
                axis,
                0,
            )
            + 1
        )

    return result


# ============================================================
# TF-IDF Query
# ============================================================

def _build_tfidf_query(
    vectorizer,
    title: str,
):

    query = (
        vectorizer.transform(
            [
                title
            ]
        )
    )

    if query.nnz == 0:

        return None

    return query


# ============================================================
# SVD Query
# ============================================================

def _build_svd_query(
    tfidf_query,
    svd,
) -> np.ndarray | None:

    reduced = (
        svd.transform(
            tfidf_query
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    norm = float(
        np.linalg.norm(
            reduced
        )
    )

    if norm < 1e-12:

        return None

    reduced = (
        normalize(
            reduced,
            norm="l2",
            axis=1,
            copy=False,
        )
    )

    return np.asarray(
        reduced[
            0
        ],
        dtype=np.float32,
    )


# ============================================================
# Chunk Scores -> Document Ranking
# ============================================================

def _collapse_to_documents(
    chunk_scores: np.ndarray,
    manifest: list[dict],
    excluded_document_id: int,
) -> list[dict]:

    best_by_document: dict[
        int,
        dict,
    ] = {}

    for (
        vector_index,
        raw_score,
    ) in enumerate(
        chunk_scores
    ):

        item = (
            manifest[
                vector_index
            ]
        )

        document_id = int(
            item[
                "document_id"
            ]
        )

        # Query paper itself must not be retrieved.
        if (
            document_id
            == excluded_document_id
        ):

            continue

        score = float(
            raw_score
        )

        current = (
            best_by_document.get(
                document_id
            )
        )

        if (
            current is None
            or score
            > current[
                "score"
            ]
        ):

            best_by_document[
                document_id
            ] = {
                "document_id": (
                    document_id
                ),

                "source": (
                    item[
                        "source"
                    ]
                ),

                "source_id": (
                    item[
                        "source_id"
                    ]
                ),

                "title": (
                    item[
                        "title"
                    ]
                ),

                "topic_axis": (
                    item[
                        "topic_axis"
                    ]
                ),

                "best_chunk_id": (
                    int(
                        item[
                            "chunk_id"
                        ]
                    )
                ),

                "best_chunk_index": (
                    int(
                        item[
                            "chunk_index"
                        ]
                    )
                ),

                "score": (
                    score
                ),
            }

    ranked = sorted(
        best_by_document.values(),
        key=lambda row: (
            row[
                "score"
            ]
        ),
        reverse=True,
    )

    return ranked


# ============================================================
# Metrics
# ============================================================

def _precision_at_k(
    relevance: list[int],
    k: int,
) -> float:

    selected = (
        relevance[
            :k
        ]
    )

    if not selected:

        return 0.0

    return float(
        sum(
            selected
        )
        / k
    )


def _recall_at_k(
    relevance: list[int],
    k: int,
    total_relevant: int,
) -> float:

    if total_relevant <= 0:

        return 0.0

    hits = sum(
        relevance[
            :k
        ]
    )

    return float(
        hits
        / total_relevant
    )


def _hit_at_1(
    relevance: list[int],
) -> float:

    if not relevance:

        return 0.0

    return float(
        relevance[
            0
        ]
    )


def _reciprocal_rank(
    relevance: list[int],
) -> float:

    for (
        rank,
        relevant,
    ) in enumerate(
        relevance,
        start=1,
    ):

        if relevant:

            return float(
                1.0
                / rank
            )

    return 0.0


def _ndcg_at_k(
    relevance: list[int],
    k: int,
    total_relevant: int,
) -> float:

    selected = (
        relevance[
            :k
        ]
    )

    dcg = 0.0

    for (
        index,
        relevant,
    ) in enumerate(
        selected
    ):

        if relevant:

            dcg += (
                1.0
                / np.log2(
                    index + 2
                )
            )

    ideal_hits = min(
        k,
        total_relevant,
    )

    if ideal_hits <= 0:

        return 0.0

    idcg = sum(
        1.0
        / np.log2(
            index + 2
        )
        for index
        in range(
            ideal_hits
        )
    )

    if idcg <= 0:

        return 0.0

    return float(
        dcg
        / idcg
    )


# ============================================================
# Evaluate One Retrieval Space
# ============================================================

def _evaluate_space(
    *,
    name: str,
    mode: str,

    documents: dict[int, dict],

    axis_counts: dict[
        str,
        int,
    ],

    manifest: list[dict],

    vectorizer,

    tfidf_matrix,

    svd=None,

    corpus_vectors=None,
) -> dict:

    print()
    print("-" * 78)

    print(
        f"[EVAL] {name}"
    )

    print("-" * 78)

    query_results = []

    skipped_queries = 0

    precision_scores = []

    recall_scores = []

    ndcg_scores = []

    mrr_scores = []

    hit1_scores = []

    axis_metric_accumulator: dict[
        str,
        dict[
            str,
            list[float],
        ],
    ] = {}

    sorted_documents = sorted(
        documents.values(),
        key=lambda row: (
            row[
                "document_id"
            ]
        ),
    )

    for (
        query_number,
        query_document,
    ) in enumerate(
        sorted_documents,
        start=1,
    ):

        document_id = int(
            query_document[
                "document_id"
            ]
        )

        title = str(
            query_document[
                "title"
            ]
        ).strip()

        query_axis = str(
            query_document[
                "topic_axis"
            ]
        )

        # Same-axis documents excluding query document.
        total_relevant = (
            axis_counts[
                query_axis
            ]
            - 1
        )

        tfidf_query = (
            _build_tfidf_query(
                vectorizer,
                title,
            )
        )

        if tfidf_query is None:

            skipped_queries += 1

            continue

        # ====================================================
        # TF-IDF
        # ====================================================

        if mode == "tfidf":

            scores = (
                tfidf_query
                @ tfidf_matrix.T
            )

            scores = (
                scores
                .toarray()[
                    0
                ]
            )

        # ====================================================
        # SVD
        # ====================================================

        elif mode == "svd":

            if (
                svd is None
                or corpus_vectors is None
            ):

                raise RuntimeError(
                    "SVD artifacts missing."
                )

            query_vector = (
                _build_svd_query(
                    tfidf_query,
                    svd,
                )
            )

            if query_vector is None:

                skipped_queries += 1

                continue

            scores = (
                corpus_vectors
                @ query_vector
            )

        else:

            raise RuntimeError(
                f"Unknown retrieval mode: "
                f"{mode}"
            )

        # ====================================================
        # Chunk -> Document
        # ====================================================

        ranked = (
            _collapse_to_documents(
                scores,
                manifest,
                document_id,
            )
        )

        relevance = [
            (
                1
                if row[
                    "topic_axis"
                ]
                == query_axis
                else 0
            )
            for row
            in ranked
        ]

        # ====================================================
        # Metrics
        # ====================================================

        precision = (
            _precision_at_k(
                relevance,
                TOP_K,
            )
        )

        recall = (
            _recall_at_k(
                relevance,
                TOP_K,
                total_relevant,
            )
        )

        ndcg = (
            _ndcg_at_k(
                relevance,
                TOP_K,
                total_relevant,
            )
        )

        reciprocal_rank = (
            _reciprocal_rank(
                relevance
            )
        )

        hit1 = (
            _hit_at_1(
                relevance
            )
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        ndcg_scores.append(
            ndcg
        )

        mrr_scores.append(
            reciprocal_rank
        )

        hit1_scores.append(
            hit1
        )

        axis_bucket = (
            axis_metric_accumulator
            .setdefault(
                query_axis,
                {
                    "precision_at_6": [],
                    "recall_at_6": [],
                    "ndcg_at_6": [],
                    "mrr": [],
                    "hit_at_1": [],
                },
            )
        )

        axis_bucket[
            "precision_at_6"
        ].append(
            precision
        )

        axis_bucket[
            "recall_at_6"
        ].append(
            recall
        )

        axis_bucket[
            "ndcg_at_6"
        ].append(
            ndcg
        )

        axis_bucket[
            "mrr"
        ].append(
            reciprocal_rank
        )

        axis_bucket[
            "hit_at_1"
        ].append(
            hit1
        )

        # ====================================================
        # Top-6 Result Detail
        # ====================================================

        top_results = []

        for (
            rank,
            row,
        ) in enumerate(
            ranked[
                :TOP_K
            ],
            start=1,
        ):

            top_results.append(
                {
                    "rank": (
                        rank
                    ),

                    "relevant_same_axis": (
                        row[
                            "topic_axis"
                        ]
                        == query_axis
                    ),

                    **row,
                }
            )

        query_results.append(
            {
                "query_document_id": (
                    document_id
                ),

                "query_source": (
                    query_document[
                        "source"
                    ]
                ),

                "query_source_id": (
                    query_document[
                        "source_id"
                    ]
                ),

                "query_title": (
                    title
                ),

                "query_axis": (
                    query_axis
                ),

                "total_relevant_documents": (
                    total_relevant
                ),

                "precision_at_6": (
                    precision
                ),

                "recall_at_6": (
                    recall
                ),

                "ndcg_at_6": (
                    ndcg
                ),

                "mrr": (
                    reciprocal_rank
                ),

                "hit_at_1": (
                    hit1
                ),

                "results": (
                    top_results
                ),
            }
        )

        if (
            query_number % 20
            == 0
        ):

            print(
                f"[EVAL] "
                f"{query_number}/"
                f"{len(sorted_documents)} "
                f"queries"
            )

    if not query_results:

        raise RuntimeError(
            f"{name}: "
            "no valid query was evaluated."
        )

    # ========================================================
    # Axis Metrics
    # ========================================================

    axis_metrics = {}

    for (
        axis,
        metrics,
    ) in (
        axis_metric_accumulator.items()
    ):

        axis_metrics[
            axis
        ] = {
            metric_name: (
                round(
                    float(
                        np.mean(
                            values
                        )
                    ),
                    6,
                )
            )
            for (
                metric_name,
                values,
            ) in metrics.items()
        }

    # ========================================================
    # Overall
    # ========================================================

    result = {
        "name": (
            name
        ),

        "evaluated_queries": (
            len(
                query_results
            )
        ),

        "skipped_queries": (
            skipped_queries
        ),

        "precision_at_6": (
            round(
                float(
                    np.mean(
                        precision_scores
                    )
                ),
                6,
            )
        ),

        "recall_at_6": (
            round(
                float(
                    np.mean(
                        recall_scores
                    )
                ),
                6,
            )
        ),

        "ndcg_at_6": (
            round(
                float(
                    np.mean(
                        ndcg_scores
                    )
                ),
                6,
            )
        ),

        "mrr": (
            round(
                float(
                    np.mean(
                        mrr_scores
                    )
                ),
                6,
            )
        ),

        "hit_at_1": (
            round(
                float(
                    np.mean(
                        hit1_scores
                    )
                ),
                6,
            )
        ),

        "axis_metrics": (
            axis_metrics
        ),

        "queries": (
            query_results
        ),
    }

    print()

    print(
        f"[EVAL] Precision@6 : "
        f"{result['precision_at_6']:.6f}"
    )

    print(
        f"[EVAL] Recall@6    : "
        f"{result['recall_at_6']:.6f}"
    )

    print(
        f"[EVAL] nDCG@6      : "
        f"{result['ndcg_at_6']:.6f}"
    )

    print(
        f"[EVAL] MRR         : "
        f"{result['mrr']:.6f}"
    )

    print(
        f"[EVAL] Hit@1       : "
        f"{result['hit_at_1']:.6f}"
    )

    return result


# ============================================================
# Select SVD Dimension
# ============================================================

def _select_svd(
    results: list[dict],
) -> dict:

    svd_results = [
        result
        for result in results
        if result[
            "name"
        ].startswith(
            "SVD-"
        )
    ]

    if not svd_results:

        raise RuntimeError(
            "No SVD evaluation result."
        )

    # ========================================================
    # Primary:
    #   nDCG@6
    #
    # Secondary:
    #   Precision@6
    #
    # Third:
    #   MRR
    #
    # Tie:
    #   smaller dimension
    # ========================================================

    ranked = sorted(
        svd_results,
        key=lambda row: (
            row[
                "ndcg_at_6"
            ],

            row[
                "precision_at_6"
            ],

            row[
                "mrr"
            ],

            -int(
                row[
                    "name"
                ].split(
                    "-"
                )[
                    1
                ]
            ),
        ),
        reverse=True,
    )

    selected = (
        ranked[
            0
        ]
    )

    dimension = int(
        selected[
            "name"
        ].split(
            "-"
        )[
            1
        ]
    )

    return {
        "dimension": (
            dimension
        ),

        "name": (
            selected[
                "name"
            ]
        ),

        "precision_at_6": (
            selected[
                "precision_at_6"
            ]
        ),

        "recall_at_6": (
            selected[
                "recall_at_6"
            ]
        ),

        "ndcg_at_6": (
            selected[
                "ndcg_at_6"
            ]
        ),

        "mrr": (
            selected[
                "mrr"
            ]
        ),

        "hit_at_1": (
            selected[
                "hit_at_1"
            ]
        ),

        "note": (
            "Automatic proxy selection. "
            "Final scientific retrieval quality "
            "must also be verified using unseen "
            "manual queries."
        ),
    }


# ============================================================
# Print Summary
# ============================================================

def _print_summary(
    results: list[dict],
    selected_svd: dict,
) -> None:

    print()
    print("=" * 94)

    print(
        "CORE-100 RETRIEVAL EVALUATION COMPLETED"
    )

    print("=" * 94)

    print(
        f"{'SPACE':<12} | "
        f"{'P@6':>9} | "
        f"{'R@6':>9} | "
        f"{'nDCG@6':>9} | "
        f"{'MRR':>9} | "
        f"{'Hit@1':>9}"
    )

    print("-" * 94)

    for result in results:

        print(
            f"{result['name']:<12} | "
            f"{result['precision_at_6']:>9.6f} | "
            f"{result['recall_at_6']:>9.6f} | "
            f"{result['ndcg_at_6']:>9.6f} | "
            f"{result['mrr']:>9.6f} | "
            f"{result['hit_at_1']:>9.6f}"
        )

    print("-" * 94)

    print()

    print(
        "Selected SVD candidate:"
    )

    print(
        f"  dimension : "
        f"{selected_svd['dimension']}"
    )

    print(
        f"  nDCG@6    : "
        f"{selected_svd['ndcg_at_6']}"
    )

    print(
        f"  P@6       : "
        f"{selected_svd['precision_at_6']}"
    )

    print(
        f"  MRR       : "
        f"{selected_svd['mrr']}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  Query document itself was excluded."
    )

    print(
        "  Chunk scores were collapsed "
        "to one score per document."
    )

    print(
        "  Relevance proxy = same topic_axis."
    )

    print(
        "  These metrics do NOT mean "
        "scientific answer correctness."
    )

    print()

    print(
        f"Report:"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print("=" * 94)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - Core-100 "
        "Retrieval Evaluation"
    )

    print("=" * 78)

    print(
        f"Evaluator version : "
        f"{EVALUATOR_VERSION}"
    )

    print(
        f"Expected docs     : "
        f"{EXPECTED_DOCUMENTS}"
    )

    print(
        f"Expected chunks   : "
        f"{EXPECTED_CHUNKS}"
    )

    print(
        f"Top-K             : "
        f"{TOP_K}"
    )

    try:

        # ====================================================
        # Preflight
        # ====================================================

        _validate_artifacts()

        print()
        print(
            "[LOAD] Loading retrieval artifacts..."
        )

        # ====================================================
        # TF-IDF
        # ====================================================

        vectorizer = (
            joblib.load(
                TFIDF_VECTORIZER_FILE
            )
        )

        tfidf_matrix = (
            sparse.load_npz(
                TFIDF_MATRIX_FILE
            )
            .tocsr()
        )

        # ====================================================
        # Manifest
        # ====================================================

        manifest = (
            _load_manifest()
        )

        documents = (
            _build_document_index(
                manifest
            )
        )

        axis_counts = (
            _axis_document_counts(
                documents
            )
        )

        # ====================================================
        # Matrix QA
        # ====================================================

        if (
            tfidf_matrix.shape[
                0
            ]
            != EXPECTED_CHUNKS
        ):

            raise RuntimeError(
                "TF-IDF row count mismatch.\n"
                f"Expected: "
                f"{EXPECTED_CHUNKS}\n"
                f"Actual  : "
                f"{tfidf_matrix.shape[0]}"
            )

        print(
            f"[LOAD] Documents : "
            f"{len(documents)}"
        )

        print(
            f"[LOAD] Chunks    : "
            f"{len(manifest)}"
        )

        print(
            f"[LOAD] TF-IDF    : "
            f"{tfidf_matrix.shape}"
        )

        print()

        print(
            "[LOAD] Document distribution:"
        )

        for (
            axis,
            count,
        ) in (
            sorted(
                axis_counts.items()
            )
        ):

            print(
                f"  {axis:22} : "
                f"{count}"
            )

        # ====================================================
        # Evaluate TF-IDF
        # ====================================================

        results = []

        tfidf_result = (
            _evaluate_space(
                name="TF-IDF",

                mode="tfidf",

                documents=(
                    documents
                ),

                axis_counts=(
                    axis_counts
                ),

                manifest=(
                    manifest
                ),

                vectorizer=(
                    vectorizer
                ),

                tfidf_matrix=(
                    tfidf_matrix
                ),
            )
        )

        results.append(
            tfidf_result
        )

        # ====================================================
        # Evaluate SVD dimensions
        # ====================================================

        for dimension in DIMENSIONS:

            print()
            print(
                f"[LOAD] Loading "
                f"SVD-{dimension}..."
            )

            svd_file = (
                CACHE_DIR
                / f"core100_svd_{dimension}.joblib"
            )

            vector_file = (
                CACHE_DIR
                / f"core100_vectors_{dimension}.npy"
            )

            svd = (
                joblib.load(
                    svd_file
                )
            )

            vectors = (
                np.load(
                    vector_file,
                    allow_pickle=False,
                )
            )

            expected_shape = (
                EXPECTED_CHUNKS,
                dimension,
            )

            if (
                vectors.shape
                != expected_shape
            ):

                raise RuntimeError(
                    "Vector shape mismatch.\n"
                    f"Dimension : "
                    f"{dimension}\n"
                    f"Expected  : "
                    f"{expected_shape}\n"
                    f"Actual    : "
                    f"{vectors.shape}"
                )

            if not (
                np.isfinite(
                    vectors
                ).all()
            ):

                raise RuntimeError(
                    f"SVD-{dimension}: "
                    "NaN/Inf detected."
                )

            result = (
                _evaluate_space(
                    name=(
                        f"SVD-{dimension}"
                    ),

                    mode="svd",

                    documents=(
                        documents
                    ),

                    axis_counts=(
                        axis_counts
                    ),

                    manifest=(
                        manifest
                    ),

                    vectorizer=(
                        vectorizer
                    ),

                    tfidf_matrix=(
                        tfidf_matrix
                    ),

                    svd=(
                        svd
                    ),

                    corpus_vectors=(
                        vectors
                    ),
                )
            )

            results.append(
                result
            )

        # ====================================================
        # Select Dimension
        # ====================================================

        selected_svd = (
            _select_svd(
                results
            )
        )

        # ====================================================
        # Report
        # ====================================================

        report = {
            "evaluator_version": (
                EVALUATOR_VERSION
            ),

            "corpus": {
                "documents": (
                    EXPECTED_DOCUMENTS
                ),

                "chunks": (
                    EXPECTED_CHUNKS
                ),

                "axis_document_counts": (
                    axis_counts
                ),
            },

            "evaluation_design": {
                "query": (
                    "core document title"
                ),

                "self_document_excluded": (
                    True
                ),

                "chunk_to_document_aggregation": (
                    "max chunk cosine score"
                ),

                "top_k": (
                    TOP_K
                ),

                "automatic_relevance_proxy": (
                    "same topic_axis"
                ),
            },

            "results": (
                results
            ),

            "selected_svd": (
                selected_svd
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
            results,
            selected_svd,
        )

        print()
        print(
            "[PASS] Retrieval evaluation completed."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "1. Confirm practical SVD dimension."
        )

        print(
            "2. Add pgvector column."
        )

        print(
            "3. Load 3113 vectors."
        )

        print(
            "4. Implement actual unseen-title "
            "Top-6 retriever."
        )

        print("=" * 78)

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "RETRIEVAL EVALUATION FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "Do NOT create pgvector "
            "schema yet."
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()