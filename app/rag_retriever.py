import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psycopg

from sklearn.preprocessing import normalize

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - Actual pgvector Retriever
# Version: core100_retriever_v3_quality_gate
# ============================================================
#
# INPUT
#
#   title
#   optional human research idea
#
#
# PIPELINE
#
#   Title
#      +
#   Human Research Idea
#        ↓
#   TF-IDF
#        ↓
#   SVD-256
#        ↓
#   L2 Normalize
#        ↓
#   PostgreSQL / pgvector
#        ↓
#   exact cosine search
#        ↓
#   candidate chunks
#        ↓
#   TEXT QUALITY GATE
#        ↓
#   document diversity
#        ↓
#   Top-6 clean evidence
#
#
# v3 quality gate additionally detects:
#
#   I I I l Global ...
#
#   g_al
#   rm/er
#   l¢lllll
#   co-plllnl
#
# and similar OCR corruption.
#
#
# IMPORTANT
#
# We reject high-confidence garbage.
# We do NOT automatically rewrite scientific evidence.
#
# ============================================================


RETRIEVER_VERSION = (
    "core100_retriever_v3_quality_gate"
)

EMBEDDING_VERSION = (
    "core100_tfidf_svd_256_v1"
)

VECTOR_DIMENSION = 256

DEFAULT_TOP_K = 6


# ============================================================
# Candidate pool
#
# Quality gate can reject candidates,
# so retrieve more than final Top-K.
# ============================================================

CANDIDATE_CHUNKS = 200


# ============================================================
# Quality configuration
# ============================================================

MIN_TEXT_CHARS = 160

MIN_WORD_COUNT = 30

MIN_ALPHA_RATIO = 0.50

MAX_WEIRD_SYMBOL_RATIO = 0.12

MIN_QUALITY_SCORE = 0.70


# Figure caption alone is weak evidence.
# Long caption + explanatory prose may survive.
FIGURE_CAPTION_MIN_WORDS = 120


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


SVD_FILE = (
    CACHE_DIR
    / "core100_svd_256.joblib"
)


RESULT_FILE = (
    REPORT_DIR
    / "last_rag_retrieval.json"
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
# Basic text cleanup
# ============================================================

def _strip_wrapping_quotes(
    text: str,
) -> str:

    text = str(
        text or ""
    ).strip()

    quote_pairs = (
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    )

    for left, right in quote_pairs:

        if (
            len(text) >= 2
            and text.startswith(left)
            and text.endswith(right)
        ):

            text = (
                text[
                    len(left):
                    len(text) - len(right)
                ]
                .strip()
            )

            break

    return text


def _normalize_spaces(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            text or ""
        ),
    ).strip()


# ============================================================
# PostgreSQL vector literal
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
# Query
# ============================================================

def _build_query_text(
    title: str,
    research_idea: str,
) -> str:

    title = (
        _strip_wrapping_quotes(
            title
        )
    )

    research_idea = (
        _strip_wrapping_quotes(
            research_idea
        )
    )

    if not title:

        raise ValueError(
            "Title must not be empty."
        )

    parts = [
        title
    ]

    if research_idea:

        parts.append(
            research_idea
        )

    return "\n".join(
        parts
    )


# ============================================================
# Query vector
# ============================================================

def _build_query_vector(
    query_text: str,
    vectorizer,
    svd,
) -> np.ndarray:

    tfidf_query = (
        vectorizer.transform(
            [
                query_text
            ]
        )
    )

    if (
        tfidf_query.nnz
        == 0
    ):

        raise RuntimeError(
            "Query produced zero TF-IDF features.\n"
            "Try a more specific scientific query."
        )

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

        raise RuntimeError(
            "Query produced a zero SVD vector."
        )

    reduced = (
        normalize(
            reduced,
            norm="l2",
            axis=1,
            copy=False,
        )
    )

    query_vector = (
        np.asarray(
            reduced[
                0
            ],
            dtype=np.float32,
        )
    )

    if (
        query_vector.shape
        != (
            VECTOR_DIMENSION,
        )
    ):

        raise RuntimeError(
            "Query dimension mismatch.\n"
            f"Expected: {VECTOR_DIMENSION}\n"
            f"Actual  : {query_vector.shape}"
        )

    return query_vector


# ============================================================
# Database validation
# ============================================================

def _validate_database(
    conn: psycopg.Connection,
) -> None:

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                COUNT(*),

                COUNT(*) FILTER (
                    WHERE embedding IS NOT NULL
                ),

                COUNT(*) FILTER (
                    WHERE embedding_version = %s
                )

            FROM public.core_chunks
            """,
            (
                EMBEDDING_VERSION,
            ),
        )

        row = (
            cur.fetchone()
        )

    (
        total_chunks,
        embedded_chunks,
        versioned_chunks,
    ) = row

    total_chunks = int(
        total_chunks
    )

    embedded_chunks = int(
        embedded_chunks
    )

    versioned_chunks = int(
        versioned_chunks
    )

    if embedded_chunks <= 0:

        raise RuntimeError(
            "No pgvector embeddings found."
        )

    if (
        embedded_chunks
        != versioned_chunks
    ):

        raise RuntimeError(
            "Embedding version mismatch.\n"
            f"Embedded : {embedded_chunks}\n"
            f"Versioned: {versioned_chunks}"
        )

    print(
        f"[DB] Total chunks    : "
        f"{total_chunks}"
    )

    print(
        f"[DB] Embedded chunks : "
        f"{embedded_chunks}"
    )


# ============================================================
# Exact cosine search
# ============================================================

def _search_candidates(
    conn: psycopg.Connection,
    query_vector: np.ndarray,
) -> list[dict]:

    vector_text = (
        _vector_literal(
            query_vector
        )
    )

    sql = """
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.section_index,
        c.section_heading,
        c.chunk_text,

        d.source,
        d.source_id,
        d.title,
        d.topic_axis,

        (
            1
            - (
                c.embedding
                <=> %s::vector
            )
        ) AS cosine_similarity

    FROM public.core_chunks c

    JOIN public.core_documents d
      ON d.id = c.document_id

    WHERE c.embedding IS NOT NULL
      AND c.embedding_version = %s

    ORDER BY
        c.embedding
        <=> %s::vector

    LIMIT %s
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                vector_text,
                EMBEDDING_VERSION,
                vector_text,
                CANDIDATE_CHUNKS,
            ),
        )

        rows = (
            cur.fetchall()
        )

    results = []

    for row in rows:

        (
            chunk_id,
            document_id,
            chunk_index,
            section_index,
            section_heading,
            chunk_text,
            source,
            source_id,
            title,
            topic_axis,
            cosine_similarity,
        ) = row

        results.append(
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

                "chunk_text": (
                    str(
                        chunk_text
                        or ""
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

                "cosine_similarity": (
                    float(
                        cosine_similarity
                    )
                ),
            }
        )

    return results


# ============================================================
# Word token extraction
# ============================================================

def _extract_word_tokens(
    text: str,
) -> list[str]:

    return re.findall(
        r"[A-Za-z]+(?:['’\-][A-Za-z]+)*",
        text,
    )


# ============================================================
# Alpha ratio
# ============================================================

def _alpha_ratio(
    text: str,
) -> float:

    visible = [
        ch
        for ch in text
        if not ch.isspace()
    ]

    if not visible:

        return 0.0

    alpha_count = sum(
        1
        for ch in visible
        if ch.isalpha()
    )

    return float(
        alpha_count
        / len(
            visible
        )
    )


# ============================================================
# Weird symbol ratio
# ============================================================

def _weird_symbol_ratio(
    text: str,
) -> float:

    allowed_punctuation = set(
        ".,;:!?%()[]{}+-/='\""
        "’“”–—<>:&"
    )

    visible = 0

    weird = 0

    for ch in text:

        if ch.isspace():

            continue

        visible += 1

        if ch.isalnum():

            continue

        if ch.isalpha():

            continue

        if ch in allowed_punctuation:

            continue

        weird += 1

    if visible == 0:

        return 1.0

    return float(
        weird
        / visible
    )


# ============================================================
# Consecutive single-letter OCR fragments
#
# Example:
#
#   I I I l Global
#
# returns 4
# ============================================================

def _max_consecutive_single_letter_run(
    text: str,
) -> int:

    raw_tokens = (
        text.split()
    )

    max_run = 0

    current_run = 0

    for raw_token in raw_tokens:

        cleaned = re.sub(
            r"^[^A-Za-z]+|[^A-Za-z]+$",
            "",
            raw_token,
        )

        if (
            len(cleaned) == 1
            and cleaned.isalpha()
        ):

            current_run += 1

            max_run = max(
                max_run,
                current_run,
            )

        else:

            current_run = 0

    return max_run


# ============================================================
# Suspicious single-letter count
# ============================================================

def _single_letter_statistics(
    tokens: list[str],
) -> tuple[int, float]:

    if not tokens:

        return (
            0,
            1.0,
        )

    suspicious = sum(
        1
        for token in tokens
        if (
            len(token) == 1
            and token.lower()
            not in {
                "a",
                "i",
            }
        )
    )

    ratio = (
        suspicious
        / len(
            tokens
        )
    )

    return (
        int(
            suspicious
        ),
        float(
            ratio
        ),
    )


# ============================================================
# Split-word OCR
#
# Examples:
#
#   T raditional
#   S pacecraft
#
# ============================================================

def _split_word_artifact_count(
    text: str,
) -> int:

    pattern = (
        r"\b"
        r"[B-HJ-Z]"
        r"\s+"
        r"[a-z]{5,}"
        r"\b"
    )

    return len(
        re.findall(
            pattern,
            text,
        )
    )


# ============================================================
# Broken-token detector
#
# Examples:
#
#   g_al
#   rm/er
#   l¢lllll
#   plllnl
#
# We count suspicious TOKENs,
# not individual characters.
# ============================================================

def _broken_ocr_token_count(
    text: str,
) -> tuple[
    int,
    list[str],
]:

    raw_tokens = (
        text.split()
    )

    suspicious_tokens = []

    for raw_token in raw_tokens:

        token = (
            raw_token.strip(
                ".,;:!?()[]{}"
                "\"'“”‘’"
            )
        )

        if not token:

            continue

        lower = (
            token.lower()
        )

        suspicious = False

        # ----------------------------------------------
        # underscore inside OCR word
        # ----------------------------------------------

        if (
            "_" in token
            and re.search(
                r"[A-Za-z]",
                token,
            )
        ):

            suspicious = True

        # ----------------------------------------------
        # obvious currency/control corruption
        # ----------------------------------------------

        if any(
            bad_character in token
            for bad_character in (
                "¢",
                "�",
                "\x00",
            )
        ):

            suspicious = True

        # ----------------------------------------------
        # letter/letter internal slash:
        #
        # rm/er
        #
        # Scientific units such as km/s are possible,
        # but repeated occurrences inside a prose chunk
        # will be handled by aggregate threshold.
        # ----------------------------------------------

        if re.search(
            r"[A-Za-z]{2,}/[A-Za-z]{2,}",
            token,
        ):

            suspicious = True

        # ----------------------------------------------
        # 3+ repeated alphabetic characters:
        #
        # plllnl
        # lllll
        # ----------------------------------------------

        if re.search(
            r"([A-Za-z])\1\1",
            lower,
        ):

            suspicious = True

        # ----------------------------------------------
        # extremely long consonant sequence
        # ----------------------------------------------

        if re.search(
            r"[bcdfghjklmnpqrstvwxyz]{7,}",
            lower,
        ):

            suspicious = True

        if suspicious:

            suspicious_tokens.append(
                token
            )

    return (
        len(
            suspicious_tokens
        ),
        suspicious_tokens,
    )


# ============================================================
# Replacement character count
# ============================================================

def _replacement_character_count(
    text: str,
) -> int:

    return (
        text.count(
            "\ufffd"
        )
        + text.count(
            "\x00"
        )
    )


# ============================================================
# Figure caption detector
# ============================================================

def _looks_like_short_figure_caption(
    text: str,
    word_count: int,
) -> bool:

    normalized = (
        text
        .lstrip()
        .lower()
    )

    figure_markers = (
        "[figure caption]",
        "figure caption",
        "[figure",
        "[fig.",
    )

    starts_as_figure = any(
        normalized.startswith(
            marker
        )
        for marker in figure_markers
    )

    return bool(
        starts_as_figure
        and word_count
        < FIGURE_CAPTION_MIN_WORDS
    )


# ============================================================
# Quality assessment
# ============================================================

def _assess_text_quality(
    text: str,
) -> dict:

    normalized = (
        _normalize_spaces(
            text
        )
    )

    tokens = (
        _extract_word_tokens(
            normalized
        )
    )

    char_count = len(
        normalized
    )

    word_count = len(
        tokens
    )

    alpha_ratio = (
        _alpha_ratio(
            normalized
        )
    )

    weird_symbol_ratio = (
        _weird_symbol_ratio(
            normalized
        )
    )

    (
        single_letter_count,
        single_letter_ratio,
    ) = (
        _single_letter_statistics(
            tokens
        )
    )

    max_single_letter_run = (
        _max_consecutive_single_letter_run(
            normalized
        )
    )

    split_word_count = (
        _split_word_artifact_count(
            normalized
        )
    )

    (
        broken_token_count,
        broken_tokens,
    ) = (
        _broken_ocr_token_count(
            normalized
        )
    )

    replacement_count = (
        _replacement_character_count(
            normalized
        )
    )

    short_figure_caption = (
        _looks_like_short_figure_caption(
            normalized,
            word_count,
        )
    )

    reasons = []

    hard_reject = False

    score = 1.0


    # ========================================================
    # 1. Minimum content
    # ========================================================

    if (
        char_count
        < MIN_TEXT_CHARS
    ):

        hard_reject = True

        reasons.append(
            "too_short_chars"
        )

    if (
        word_count
        < MIN_WORD_COUNT
    ):

        hard_reject = True

        reasons.append(
            "too_few_words"
        )


    # ========================================================
    # 2. Alphabetic density
    # ========================================================

    if (
        alpha_ratio
        < MIN_ALPHA_RATIO
    ):

        hard_reject = True

        reasons.append(
            "low_alpha_ratio"
        )

    elif (
        alpha_ratio
        < 0.60
    ):

        score -= 0.10

        reasons.append(
            "moderately_low_alpha_ratio"
        )


    # ========================================================
    # 3. Consecutive single-letter OCR
    #
    # Critical fix for:
    #
    #   I I I l Global ...
    # ========================================================

    if (
        max_single_letter_run
        >= 3
    ):

        hard_reject = True

        reasons.append(
            "consecutive_single_letter_ocr"
        )


    # ========================================================
    # 4. Overall single-letter noise
    # ========================================================

    if (
        single_letter_count
        >= 6
        and single_letter_ratio
        >= 0.05
    ):

        hard_reject = True

        reasons.append(
            "excessive_single_letter_fragments"
        )

    elif (
        single_letter_count
        >= 3
        and single_letter_ratio
        >= 0.03
    ):

        score -= 0.10

        reasons.append(
            "single_letter_noise"
        )


    # ========================================================
    # 5. Split words
    #
    # T raditional
    # ========================================================

    if (
        split_word_count
        >= 3
    ):

        hard_reject = True

        reasons.append(
            "multiple_split_word_artifacts"
        )

    elif (
        split_word_count
        >= 1
    ):

        score -= (
            0.10
            * split_word_count
        )

        reasons.append(
            "split_word_artifact"
        )


    # ========================================================
    # 6. Broken OCR tokens
    #
    # g_al / rm/er / l¢lllll / plllnl
    # ========================================================

    if (
        broken_token_count
        >= 2
    ):

        hard_reject = True

        reasons.append(
            "multiple_broken_ocr_tokens"
        )

    elif (
        broken_token_count
        == 1
    ):

        score -= 0.15

        reasons.append(
            "broken_ocr_token"
        )


    # ========================================================
    # 7. Weird symbols
    # ========================================================

    if (
        weird_symbol_ratio
        > MAX_WEIRD_SYMBOL_RATIO
    ):

        hard_reject = True

        reasons.append(
            "excessive_symbol_noise"
        )

    elif (
        weird_symbol_ratio
        > 0.05
    ):

        score -= 0.10

        reasons.append(
            "symbol_noise"
        )


    # ========================================================
    # 8. Replacement characters
    # ========================================================

    if (
        replacement_count
        > 0
    ):

        hard_reject = True

        reasons.append(
            "replacement_character"
        )


    # ========================================================
    # 9. Figure-only chunk
    # ========================================================

    if short_figure_caption:

        hard_reject = True

        reasons.append(
            "short_figure_caption"
        )


    # ========================================================
    # 10. Clamp score
    # ========================================================

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


    accepted = (
        not hard_reject
        and score
        >= MIN_QUALITY_SCORE
    )


    if (
        not hard_reject
        and score
        < MIN_QUALITY_SCORE
    ):

        reasons.append(
            "quality_score_below_threshold"
        )


    return {
        "accepted": (
            bool(
                accepted
            )
        ),

        "quality_score": (
            round(
                float(
                    score
                ),
                6,
            )
        ),

        "char_count": (
            int(
                char_count
            )
        ),

        "word_count": (
            int(
                word_count
            )
        ),

        "alpha_ratio": (
            round(
                float(
                    alpha_ratio
                ),
                6,
            )
        ),

        "weird_symbol_ratio": (
            round(
                float(
                    weird_symbol_ratio
                ),
                6,
            )
        ),

        "single_letter_count": (
            int(
                single_letter_count
            )
        ),

        "single_letter_ratio": (
            round(
                float(
                    single_letter_ratio
                ),
                6,
            )
        ),

        "max_single_letter_run": (
            int(
                max_single_letter_run
            )
        ),

        "split_word_artifacts": (
            int(
                split_word_count
            )
        ),

        "broken_ocr_token_count": (
            int(
                broken_token_count
            )
        ),

        "broken_ocr_tokens": (
            broken_tokens[
                :10
            ]
        ),

        "replacement_characters": (
            int(
                replacement_count
            )
        ),

        "short_figure_caption": (
            bool(
                short_figure_caption
            )
        ),

        "reasons": (
            reasons
        ),
    }


# ============================================================
# Apply quality gate
# ============================================================

def _apply_quality_gate(
    candidates: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:

    accepted = []

    rejected = []

    for candidate in candidates:

        quality = (
            _assess_text_quality(
                candidate[
                    "chunk_text"
                ]
            )
        )

        item = {
            **candidate,

            "quality": (
                quality
            ),
        }

        if quality[
            "accepted"
        ]:

            accepted.append(
                item
            )

        else:

            rejected.append(
                item
            )

    return (
        accepted,
        rejected,
    )


# ============================================================
# Quality summary
# ============================================================

def _quality_summary(
    accepted: list[dict],
    rejected: list[dict],
) -> dict:

    counter = Counter()

    for item in rejected:

        for reason in (
            item[
                "quality"
            ][
                "reasons"
            ]
        ):

            counter[
                reason
            ] += 1

    return {
        "candidate_count": (
            len(
                accepted
            )
            + len(
                rejected
            )
        ),

        "accepted_count": (
            len(
                accepted
            )
        ),

        "rejected_count": (
            len(
                rejected
            )
        ),

        "rejection_reasons": (
            dict(
                counter
            )
        ),
    }


# ============================================================
# Select diverse Top-K
# ============================================================

def _select_top_evidence(
    accepted_candidates: list[dict],
    top_k: int,
) -> list[dict]:

    selected = []

    seen_documents = set()

    for candidate in accepted_candidates:

        document_id = (
            candidate[
                "document_id"
            ]
        )

        if (
            document_id
            in seen_documents
        ):

            continue

        seen_documents.add(
            document_id
        )

        selected.append(
            candidate
        )

        if (
            len(
                selected
            )
            >= top_k
        ):

            break

    if (
        len(
            selected
        )
        < top_k
    ):

        raise RuntimeError(
            "Not enough unique clean documents "
            "after quality filtering.\n"
            f"Required: {top_k}\n"
            f"Found   : {len(selected)}"
        )

    for (
        rank,
        item,
    ) in enumerate(
        selected,
        start=1,
    ):

        item[
            "rank"
        ] = rank

    return selected


# ============================================================
# Rejected candidate sample
# ============================================================

def _print_rejected_sample(
    rejected: list[dict],
    limit: int = 8,
) -> None:

    if not rejected:

        print(
            "[QUALITY] No candidate rejected."
        )

        return

    print()
    print(
        "[QUALITY] Rejected candidate sample:"
    )

    for item in (
        rejected[
            :limit
        ]
    ):

        quality = (
            item[
                "quality"
            ]
        )

        preview = (
            _normalize_spaces(
                item[
                    "chunk_text"
                ]
            )
        )

        if (
            len(
                preview
            )
            > 220
        ):

            preview = (
                preview[
                    :220
                ]
                + "..."
            )

        print()

        print(
            f"  chunk_id="
            f"{item['chunk_id']} "
            f"document_id="
            f"{item['document_id']} "
            f"cosine="
            f"{item['cosine_similarity']:.6f}"
        )

        print(
            f"  quality="
            f"{quality['quality_score']:.3f}"
        )

        print(
            f"  reasons="
            f"{quality['reasons']}"
        )

        if (
            quality[
                "broken_ocr_tokens"
            ]
        ):

            print(
                f"  broken_tokens="
                f"{quality['broken_ocr_tokens']}"
            )

        print(
            f"  preview="
            f"{preview}"
        )


# ============================================================
# Print final Top-K
# ============================================================

def _print_results(
    title: str,
    research_idea: str,
    evidence: list[dict],
    quality_summary: dict,
) -> None:

    print()
    print("=" * 100)

    print(
        "TEAM B - TOP-6 CLEAN RAG EVIDENCE"
    )

    print("=" * 100)

    print(
        "Title:"
    )

    print(
        f"  {title}"
    )

    if research_idea:

        print()

        print(
            "Human Research Idea:"
        )

        print(
            f"  {research_idea}"
        )

    print()

    print(
        "Quality Gate:"
    )

    print(
        f"  Candidates : "
        f"{quality_summary['candidate_count']}"
    )

    print(
        f"  Accepted   : "
        f"{quality_summary['accepted_count']}"
    )

    print(
        f"  Rejected   : "
        f"{quality_summary['rejected_count']}"
    )

    print()
    print("-" * 100)

    for item in evidence:

        quality = (
            item[
                "quality"
            ]
        )

        print()

        print(
            f"[{item['rank']}] "
            f"{item['title']}"
        )

        print(
            f"    Source     : "
            f"{item['source']} / "
            f"{item['source_id']}"
        )

        print(
            f"    Topic      : "
            f"{item['topic_axis']}"
        )

        print(
            f"    Section    : "
            f"{item['section_heading']}"
        )

        print(
            f"    Chunk      : "
            f"{item['chunk_index']}"
        )

        print(
            f"    Cosine     : "
            f"{item['cosine_similarity']:.6f}"
        )

        print(
            f"    Quality    : "
            f"{quality['quality_score']:.3f}"
        )

        preview = (
            _normalize_spaces(
                item[
                    "chunk_text"
                ]
            )
        )

        if (
            len(
                preview
            )
            > 600
        ):

            preview = (
                preview[
                    :600
                ]
                + "..."
            )

        print(
            f"    Evidence   : "
            f"{preview}"
        )

    print()
    print("=" * 100)


# ============================================================
# Retrieval with diagnostics
# ============================================================

def retrieve_with_diagnostics(
    title: str,
    research_idea: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> dict:

    if not (
        TFIDF_VECTORIZER_FILE.exists()
    ):

        raise FileNotFoundError(
            TFIDF_VECTORIZER_FILE
        )

    if not (
        SVD_FILE.exists()
    ):

        raise FileNotFoundError(
            SVD_FILE
        )

    vectorizer = (
        joblib.load(
            TFIDF_VECTORIZER_FILE
        )
    )

    svd = (
        joblib.load(
            SVD_FILE
        )
    )

    title = (
        _strip_wrapping_quotes(
            title
        )
    )

    research_idea = (
        _strip_wrapping_quotes(
            research_idea
        )
    )

    query_text = (
        _build_query_text(
            title,
            research_idea,
        )
    )

    query_vector = (
        _build_query_vector(
            query_text,
            vectorizer,
            svd,
        )
    )

    with psycopg.connect(
        settings.dsn
    ) as conn:

        _validate_database(
            conn
        )

        candidates = (
            _search_candidates(
                conn,
                query_vector,
            )
        )

    (
        accepted_candidates,
        rejected_candidates,
    ) = (
        _apply_quality_gate(
            candidates
        )
    )

    summary = (
        _quality_summary(
            accepted_candidates,
            rejected_candidates,
        )
    )

    evidence = (
        _select_top_evidence(
            accepted_candidates,
            top_k,
        )
    )

    return {
        "title": (
            title
        ),

        "research_idea": (
            research_idea
        ),

        "evidence": (
            evidence
        ),

        "quality_summary": (
            summary
        ),

        "rejected_candidates": (
            rejected_candidates
        ),
    }


# ============================================================
# Backward compatible API
# ============================================================

def retrieve(
    title: str,
    research_idea: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:

    result = (
        retrieve_with_diagnostics(
            title=title,
            research_idea=research_idea,
            top_k=top_k,
        )
    )

    return (
        result[
            "evidence"
        ]
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "TEAM B Core-100 "
                "pgvector Top-6 retriever "
                "with OCR/evidence quality gate"
            )
        )
    )

    parser.add_argument(
        "--title",
        type=str,
        default="",
    )

    parser.add_argument(
        "--idea",
        type=str,
        default="",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--show-rejected",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    title = (
        args.title.strip()
    )

    research_idea = (
        args.idea.strip()
    )

    if not title:

        print()
        print(
            "Enter research title:"
        )

        title = (
            input(
                "> "
            )
            .strip()
        )

    if not research_idea:

        print()
        print(
            "Enter human research idea "
            "(optional, press Enter to skip):"
        )

        research_idea = (
            input(
                "> "
            )
            .strip()
        )

    title = (
        _strip_wrapping_quotes(
            title
        )
    )

    research_idea = (
        _strip_wrapping_quotes(
            research_idea
        )
    )

    print()
    print("=" * 78)

    print(
        "TEAM B - Core-100 "
        "pgvector Retriever"
    )

    print("=" * 78)

    print(
        f"Retriever version : "
        f"{RETRIEVER_VERSION}"
    )

    print(
        f"Vector dimension  : "
        f"{VECTOR_DIMENSION}"
    )

    print(
        f"Embedding version : "
        f"{EMBEDDING_VERSION}"
    )

    print(
        f"Candidate chunks  : "
        f"{CANDIDATE_CHUNKS}"
    )

    try:

        result = (
            retrieve_with_diagnostics(
                title=title,
                research_idea=research_idea,
                top_k=args.top_k,
            )
        )

        title = (
            result[
                "title"
            ]
        )

        research_idea = (
            result[
                "research_idea"
            ]
        )

        evidence = (
            result[
                "evidence"
            ]
        )

        quality_summary = (
            result[
                "quality_summary"
            ]
        )

        rejected_candidates = (
            result[
                "rejected_candidates"
            ]
        )

        print()
        print(
            "[QUALITY] "
            f"Candidates : "
            f"{quality_summary['candidate_count']}"
        )

        print(
            "[QUALITY] "
            f"Accepted   : "
            f"{quality_summary['accepted_count']}"
        )

        print(
            "[QUALITY] "
            f"Rejected   : "
            f"{quality_summary['rejected_count']}"
        )

        if (
            quality_summary[
                "rejection_reasons"
            ]
        ):

            print(
                "[QUALITY] "
                "Rejection reasons:"
            )

            for (
                reason,
                count,
            ) in (
                quality_summary[
                    "rejection_reasons"
                ].items()
            ):

                print(
                    f"  {reason:<38} "
                    f"{count}"
                )

        if (
            args.show_rejected
        ):

            _print_rejected_sample(
                rejected_candidates,
            )

        _print_results(
            title,
            research_idea,
            evidence,
            quality_summary,
        )

        payload = {
            "retriever_version": (
                RETRIEVER_VERSION
            ),

            "embedding_version": (
                EMBEDDING_VERSION
            ),

            "vector_dimension": (
                VECTOR_DIMENSION
            ),

            "candidate_chunks": (
                CANDIDATE_CHUNKS
            ),

            "title": (
                title
            ),

            "research_idea": (
                research_idea
            ),

            "top_k": (
                args.top_k
            ),

            "quality_gate": (
                quality_summary
            ),

            "evidence": (
                evidence
            ),

            "rejected_candidates": [
                {
                    "chunk_id": (
                        item[
                            "chunk_id"
                        ]
                    ),

                    "document_id": (
                        item[
                            "document_id"
                        ]
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

                    "chunk_index": (
                        item[
                            "chunk_index"
                        ]
                    ),

                    "cosine_similarity": (
                        item[
                            "cosine_similarity"
                        ]
                    ),

                    "quality": (
                        item[
                            "quality"
                        ]
                    ),
                }

                for item
                in rejected_candidates
            ],
        }

        _save_json(
            RESULT_FILE,
            payload,
        )

        print()
        print(
            f"[PASS] Top-{args.top_k} "
            f"clean retrieval completed."
        )

        print(
            f"[REPORT] "
            f"{RESULT_FILE}"
        )

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "RAG RETRIEVAL FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()