import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - Stage 1 Smoke Corpus Preparation
# Version: stage1_smoke_prepare_v1
# ============================================================
#
# PURPOSE
#
# 현재 Core-100을 최종 학습 corpus로 쓰는 것이 아니다.
#
# 목적:
#
#   Core-100
#      ↓
#   Stage-1 training text 생성
#      ↓
#   자체 tokenizer
#      ↓
#   Decoder-only Transformer
#      ↓
#   next-token training
#      ↓
#   checkpoint
#      ↓
#   generation
#
# 전체 파이프라인이 실제로 동작하는지 먼저 검증한다.
#
#
# 최종 확장:
#
#   smoke 100
#      ↓
#   20K
#      ↓
#   50K
#      ↓
#   100K
#      ↓
#   300K+
#
#
# IMPORTANT
#
# - RAG용 core_chunks를 학습 corpus로 쓰지 않는다.
# - document-level clean_content를 사용한다.
# - train/validation은 document 단위로 분리한다.
# - 같은 논문의 일부가 train과 val에 동시에 들어가지 않는다.
# ============================================================


PREP_VERSION = "stage1_smoke_prepare_v1"

SOURCE_TABLE = "public.core_documents"

EXPECTED_DOCUMENTS = 100

RANDOM_SEED = 42

VALIDATION_RATIO = 0.10

MIN_DOCUMENT_CHARS = 1000


# ============================================================
# Output Paths
# ============================================================

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)


TRAIN_JSONL = (
    PROCESSED_DIR
    / "stage1_smoke_train.jsonl"
)


VAL_JSONL = (
    PROCESSED_DIR
    / "stage1_smoke_val.jsonl"
)


TRAIN_TEXT = (
    PROCESSED_DIR
    / "stage1_smoke_train.txt"
)


VAL_TEXT = (
    PROCESSED_DIR
    / "stage1_smoke_val.txt"
)


REPORT_FILE = (
    REPORT_DIR
    / "stage1_smoke_prepare_report.json"
)


# ============================================================
# Structural Tokens
# ============================================================

BOS_TOKEN = "<|bos|>"

EOS_TOKEN = "<|eos|>"

TITLE_TOKEN = "<|title|>"

BODY_TOKEN = "<|body|>"


# ============================================================
# JSON Helpers
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


def _write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            f.write(
                "\n"
            )


# ============================================================
# DB Schema Validation
# ============================================================

def _validate_schema(
    conn: psycopg.Connection,
) -> None:

    sql = """
    SELECT
        column_name

    FROM information_schema.columns

    WHERE table_schema = 'public'
      AND table_name = 'core_documents'

    ORDER BY ordinal_position
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
            row[
                0
            ]
        )
        for row in rows
    }

    required_columns = {
        "id",
        "source",
        "source_id",
        "title",
        "topic_axis",
        "clean_content",
    }

    missing = (
        required_columns
        - columns
    )

    if missing:

        raise RuntimeError(
            "core_documents schema is missing "
            "required columns.\n"
            f"Missing   : "
            f"{sorted(missing)}\n"
            f"Available : "
            f"{sorted(columns)}"
        )


# ============================================================
# Load Core Documents
# ============================================================

def _load_documents(
    conn: psycopg.Connection,
) -> list[dict]:

    sql = f"""
    SELECT
        id,
        source,
        source_id,
        title,
        topic_axis,
        clean_content

    FROM {SOURCE_TABLE}

    ORDER BY id
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = (
            cur.fetchall()
        )

    documents = []

    for row in rows:

        (
            document_id,
            source,
            source_id,
            title,
            topic_axis,
            clean_content,
        ) = row

        documents.append(
            {
                "document_id": (
                    int(
                        document_id
                    )
                ),

                "source": (
                    str(
                        source
                        or ""
                    )
                    .strip()
                ),

                "source_id": (
                    str(
                        source_id
                        or ""
                    )
                    .strip()
                ),

                "title": (
                    str(
                        title
                        or ""
                    )
                    .strip()
                ),

                "topic_axis": (
                    str(
                        topic_axis
                        or ""
                    )
                    .strip()
                ),

                "text": (
                    str(
                        clean_content
                        or ""
                    )
                ),
            }
        )

    return documents


# ============================================================
# OCR Garbage Detection
# ============================================================

def _max_single_letter_run(
    line: str,
) -> int:

    raw_tokens = (
        line.split()
    )

    current_run = 0

    max_run = 0

    for raw_token in raw_tokens:

        token = re.sub(
            r"^[^A-Za-z]+|[^A-Za-z]+$",
            "",
            raw_token,
        )

        if (
            len(
                token
            )
            == 1
            and token.isalpha()
        ):

            current_run += 1

            max_run = max(
                max_run,
                current_run,
            )

        else:

            current_run = 0

    return max_run


def _broken_token_count(
    line: str,
) -> int:

    count = 0

    for raw_token in (
        line.split()
    ):

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


        # ====================================================
        # underscore corruption
        #
        # g_al
        # ====================================================

        if (
            "_" in token
            and re.search(
                r"[A-Za-z]",
                token,
            )
        ):

            suspicious = True


        # ====================================================
        # broken / replacement chars
        # ====================================================

        if any(
            char in token
            for char in (
                "¢",
                "�",
                "\x00",
            )
        ):

            suspicious = True


        # ====================================================
        # repeated OCR characters
        #
        # plllnl
        # lllll
        # ====================================================

        if re.search(
            r"([A-Za-z])\1\1",
            lower,
        ):

            suspicious = True


        # ====================================================
        # extreme consonant run
        # ====================================================

        if re.search(
            r"[bcdfghjklmnpqrstvwxyz]{7,}",
            lower,
        ):

            suspicious = True


        if suspicious:

            count += 1

    return count


def _is_obvious_garbage_line(
    line: str,
) -> bool:

    stripped = (
        line.strip()
    )

    if not stripped:

        return False


    # ========================================================
    # Replacement / null chars
    # ========================================================

    if (
        "�"
        in stripped
        or "\x00"
        in stripped
    ):

        return True


    # ========================================================
    # OCR spaced-letter corruption
    #
    # I I I l Global ...
    # ========================================================

    if (
        _max_single_letter_run(
            stripped
        )
        >= 4
    ):

        return True


    # ========================================================
    # Multiple obviously broken tokens
    # ========================================================

    if (
        _broken_token_count(
            stripped
        )
        >= 4
    ):

        return True


    # ========================================================
    # Very low alphabetic density
    # ========================================================

    visible = [
        char
        for char in stripped
        if not char.isspace()
    ]

    if (
        len(
            visible
        )
        >= 80
    ):

        alpha_count = sum(
            1
            for char in visible
            if char.isalpha()
        )

        alpha_ratio = (
            alpha_count
            / len(
                visible
            )
        )

        if (
            alpha_ratio
            < 0.30
        ):

            return True

    return False


# ============================================================
# Normalize One Document
# ============================================================

def _normalize_document(
    text: str,
) -> tuple[
    str,
    int,
]:

    text = (
        unicodedata.normalize(
            "NFC",
            str(
                text
                or ""
            ),
        )
    )

    text = (
        text.replace(
            "\x00",
            ""
        )
    )

    raw_lines = (
        text.splitlines()
    )

    cleaned_lines = []

    dropped_lines = 0


    for raw_line in raw_lines:

        line = (
            re.sub(
                r"[ \t]+",
                " ",
                raw_line,
            )
            .strip()
        )


        # ====================================================
        # Blank line
        # ====================================================

        if not line:

            if (
                cleaned_lines
                and cleaned_lines[
                    -1
                ]
                != ""
            ):

                cleaned_lines.append(
                    ""
                )

            continue


        # ====================================================
        # Garbage line
        # ====================================================

        if (
            _is_obvious_garbage_line(
                line
            )
        ):

            dropped_lines += 1

            continue


        # ====================================================
        # Keep line
        # ====================================================

        cleaned_lines.append(
            line
        )


    normalized = (
        "\n".join(
            cleaned_lines
        )
        .strip()
    )


    # ========================================================
    # Remove excessive blank lines
    # ========================================================

    normalized = re.sub(
        r"\n{3,}",
        "\n\n",
        normalized,
    )

    return (
        normalized,
        dropped_lines,
    )


# ============================================================
# Prepare All Documents
# ============================================================

def _prepare_documents(
    documents: list[dict],
) -> tuple[
    list[dict],
    dict,
]:

    prepared = []

    rejected = []

    total_dropped_lines = 0

    seen_hashes = set()

    duplicate_count = 0


    for document in documents:

        (
            normalized_text,
            dropped_lines,
        ) = (
            _normalize_document(
                document[
                    "text"
                ]
            )
        )

        total_dropped_lines += (
            dropped_lines
        )


        # ====================================================
        # Minimum length
        # ====================================================

        if (
            len(
                normalized_text
            )
            < MIN_DOCUMENT_CHARS
        ):

            rejected.append(
                {
                    "document_id": (
                        document[
                            "document_id"
                        ]
                    ),

                    "source": (
                        document[
                            "source"
                        ]
                    ),

                    "source_id": (
                        document[
                            "source_id"
                        ]
                    ),

                    "title": (
                        document[
                            "title"
                        ]
                    ),

                    "reason": (
                        "too_short_after_cleanup"
                    ),

                    "char_count": (
                        len(
                            normalized_text
                        )
                    ),
                }
            )

            continue


        # ====================================================
        # Exact document duplicate check
        # ====================================================

        content_hash = (
            hashlib.sha256(
                normalized_text.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        if (
            content_hash
            in seen_hashes
        ):

            duplicate_count += 1

            rejected.append(
                {
                    "document_id": (
                        document[
                            "document_id"
                        ]
                    ),

                    "source": (
                        document[
                            "source"
                        ]
                    ),

                    "source_id": (
                        document[
                            "source_id"
                        ]
                    ),

                    "title": (
                        document[
                            "title"
                        ]
                    ),

                    "reason": (
                        "exact_duplicate_document"
                    ),

                    "content_hash": (
                        content_hash
                    ),
                }
            )

            continue

        seen_hashes.add(
            content_hash
        )


        # ====================================================
        # Accept
        # ====================================================

        word_count = len(
            normalized_text.split()
        )

        prepared.append(
            {
                "document_id": (
                    document[
                        "document_id"
                    ]
                ),

                "source": (
                    document[
                        "source"
                    ]
                ),

                "source_id": (
                    document[
                        "source_id"
                    ]
                ),

                "title": (
                    document[
                        "title"
                    ]
                ),

                "topic_axis": (
                    document[
                        "topic_axis"
                    ]
                ),

                "text": (
                    normalized_text
                ),

                "char_count": (
                    len(
                        normalized_text
                    )
                ),

                "word_count": (
                    word_count
                ),

                "content_hash": (
                    content_hash
                ),

                "dropped_garbage_lines": (
                    dropped_lines
                ),
            }
        )


    stats = {
        "input_documents": (
            len(
                documents
            )
        ),

        "prepared_documents": (
            len(
                prepared
            )
        ),

        "rejected_documents": (
            len(
                rejected
            )
        ),

        "duplicate_documents": (
            duplicate_count
        ),

        "dropped_garbage_lines": (
            total_dropped_lines
        ),

        "rejected": (
            rejected
        ),
    }

    return (
        prepared,
        stats,
    )


# ============================================================
# Stratified Train / Validation Split
# ============================================================

def _split_documents(
    documents: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:

    rng = (
        random.Random(
            RANDOM_SEED
        )
    )

    by_axis: dict[
        str,
        list[dict],
    ] = {}


    for document in documents:

        axis = (
            document[
                "topic_axis"
            ]
        )

        by_axis.setdefault(
            axis,
            [],
        ).append(
            document
        )


    train_documents = []

    validation_documents = []


    for (
        axis,
        group,
    ) in sorted(
        by_axis.items()
    ):

        group = (
            list(
                group
            )
        )

        rng.shuffle(
            group
        )

        validation_count = max(
            1,
            int(
                round(
                    len(
                        group
                    )
                    * VALIDATION_RATIO
                )
            ),
        )

        validation_documents.extend(
            group[
                :validation_count
            ]
        )

        train_documents.extend(
            group[
                validation_count:
            ]
        )


    # ========================================================
    # deterministic output
    # ========================================================

    train_documents.sort(
        key=lambda row: (
            row[
                "document_id"
            ]
        )
    )

    validation_documents.sort(
        key=lambda row: (
            row[
                "document_id"
            ]
        )
    )


    return (
        train_documents,
        validation_documents,
    )


# ============================================================
# Language Model Text Format
# ============================================================

def _to_lm_text(
    document: dict,
) -> str:

    return (
        f"{BOS_TOKEN}\n"
        f"{TITLE_TOKEN}\n"
        f"{document['title']}\n"
        f"{BODY_TOKEN}\n"
        f"{document['text']}\n"
        f"{EOS_TOKEN}"
    )


# ============================================================
# Write Plain Text Corpus
# ============================================================

def _write_lm_text(
    path: Path,
    documents: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        for (
            index,
            document,
        ) in enumerate(
            documents
        ):

            if index:

                f.write(
                    "\n\n"
                )

            f.write(
                _to_lm_text(
                    document
                )
            )


# ============================================================
# Statistics
# ============================================================

def _document_stats(
    documents: list[dict],
) -> dict:

    if not documents:

        return {
            "documents": 0,
            "characters_total": 0,
            "words_total": 0,
            "characters_avg": 0,
            "words_avg": 0,
            "axis_counts": {},
            "source_counts": {},
        }


    char_counts = [
        document[
            "char_count"
        ]
        for document in documents
    ]

    word_counts = [
        document[
            "word_count"
        ]
        for document in documents
    ]


    axis_counts = Counter(
        document[
            "topic_axis"
        ]
        for document in documents
    )


    source_counts = Counter(
        document[
            "source"
        ]
        for document in documents
    )


    return {
        "documents": (
            len(
                documents
            )
        ),

        "characters_total": (
            int(
                sum(
                    char_counts
                )
            )
        ),

        "words_total": (
            int(
                sum(
                    word_counts
                )
            )
        ),

        "characters_avg": (
            round(
                float(
                    sum(
                        char_counts
                    )
                    / len(
                        char_counts
                    )
                ),
                2,
            )
        ),

        "words_avg": (
            round(
                float(
                    sum(
                        word_counts
                    )
                    / len(
                        word_counts
                    )
                ),
                2,
            )
        ),

        "axis_counts": (
            dict(
                axis_counts
            )
        ),

        "source_counts": (
            dict(
                source_counts
            )
        ),
    }


# ============================================================
# Verify Split
# ============================================================

def _verify_split(
    train_documents: list[dict],
    validation_documents: list[dict],
) -> None:

    train_ids = {
        document[
            "document_id"
        ]
        for document in train_documents
    }

    validation_ids = {
        document[
            "document_id"
        ]
        for document
        in validation_documents
    }


    overlap = (
        train_ids
        & validation_ids
    )

    if overlap:

        raise RuntimeError(
            "Train/validation document leakage detected.\n"
            f"Overlapping document IDs: "
            f"{sorted(overlap)}"
        )


    train_hashes = {
        document[
            "content_hash"
        ]
        for document in train_documents
    }

    validation_hashes = {
        document[
            "content_hash"
        ]
        for document
        in validation_documents
    }


    hash_overlap = (
        train_hashes
        & validation_hashes
    )

    if hash_overlap:

        raise RuntimeError(
            "Train/validation content leakage detected.\n"
            f"Duplicate hashes: "
            f"{len(hash_overlap)}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - Stage 1 "
        "Smoke Corpus Preparation"
    )

    print("=" * 78)

    print(
        f"Version           : "
        f"{PREP_VERSION}"
    )

    print(
        f"Source table      : "
        f"{SOURCE_TABLE}"
    )

    print(
        f"Expected docs     : "
        f"{EXPECTED_DOCUMENTS}"
    )

    print(
        f"Validation ratio  : "
        f"{VALIDATION_RATIO}"
    )

    print(
        f"Random seed       : "
        f"{RANDOM_SEED}"
    )


    try:

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
                "[DB] Schema OK."
            )

            raw_documents = (
                _load_documents(
                    conn
                )
            )


        print(
            f"[DB] Loaded documents: "
            f"{len(raw_documents)}"
        )


        if (
            len(
                raw_documents
            )
            != EXPECTED_DOCUMENTS
        ):

            raise RuntimeError(
                "Core-100 document count mismatch.\n"
                f"Expected: "
                f"{EXPECTED_DOCUMENTS}\n"
                f"Actual  : "
                f"{len(raw_documents)}"
            )


        # ====================================================
        # Cleaning
        # ====================================================

        print()
        print(
            "[CLEAN] Preparing "
            "Stage-1 scientific text..."
        )


        (
            prepared_documents,
            cleaning_stats,
        ) = (
            _prepare_documents(
                raw_documents
            )
        )


        print(
            f"[CLEAN] Input docs     : "
            f"{cleaning_stats['input_documents']}"
        )

        print(
            f"[CLEAN] Prepared docs  : "
            f"{cleaning_stats['prepared_documents']}"
        )

        print(
            f"[CLEAN] Rejected docs  : "
            f"{cleaning_stats['rejected_documents']}"
        )

        print(
            f"[CLEAN] Duplicate docs : "
            f"{cleaning_stats['duplicate_documents']}"
        )

        print(
            f"[CLEAN] Garbage lines  : "
            f"{cleaning_stats['dropped_garbage_lines']}"
        )


        # ====================================================
        # Smoke corpus threshold
        # ====================================================

        if (
            len(
                prepared_documents
            )
            < 90
        ):

            raise RuntimeError(
                "Too many documents were rejected "
                "during Stage-1 preparation.\n"
                f"Prepared: "
                f"{len(prepared_documents)}\n"
                f"Expected at least: 90"
            )


        # ====================================================
        # Split
        # ====================================================

        (
            train_documents,
            validation_documents,
        ) = (
            _split_documents(
                prepared_documents
            )
        )


        _verify_split(
            train_documents,
            validation_documents,
        )


        print()
        print(
            "[SPLIT] Document-level split PASS."
        )

        print(
            f"[SPLIT] Train docs : "
            f"{len(train_documents)}"
        )

        print(
            f"[SPLIT] Val docs   : "
            f"{len(validation_documents)}"
        )


        # ====================================================
        # Save
        # ====================================================

        _write_jsonl(
            TRAIN_JSONL,
            train_documents,
        )

        _write_jsonl(
            VAL_JSONL,
            validation_documents,
        )


        _write_lm_text(
            TRAIN_TEXT,
            train_documents,
        )

        _write_lm_text(
            VAL_TEXT,
            validation_documents,
        )


        # ====================================================
        # Stats
        # ====================================================

        train_stats = (
            _document_stats(
                train_documents
            )
        )

        validation_stats = (
            _document_stats(
                validation_documents
            )
        )


        # ====================================================
        # Report
        # ====================================================

        report = {
            "prep_version": (
                PREP_VERSION
            ),

            "purpose": (
                "Core-100 smoke corpus for "
                "validating TEAM B direct "
                "Transformer training pipeline. "
                "This is NOT the final Stage-1 corpus."
            ),

            "source_table": (
                SOURCE_TABLE
            ),

            "expected_documents": (
                EXPECTED_DOCUMENTS
            ),

            "random_seed": (
                RANDOM_SEED
            ),

            "validation_ratio": (
                VALIDATION_RATIO
            ),

            "minimum_document_characters": (
                MIN_DOCUMENT_CHARS
            ),

            "special_tokens": {
                "bos": (
                    BOS_TOKEN
                ),

                "eos": (
                    EOS_TOKEN
                ),

                "title": (
                    TITLE_TOKEN
                ),

                "body": (
                    BODY_TOKEN
                ),
            },

            "cleaning": (
                cleaning_stats
            ),

            "train": (
                train_stats
            ),

            "validation": (
                validation_stats
            ),

            "files": {
                "train_jsonl": (
                    str(
                        TRAIN_JSONL
                    )
                ),

                "val_jsonl": (
                    str(
                        VAL_JSONL
                    )
                ),

                "train_text": (
                    str(
                        TRAIN_TEXT
                    )
                ),

                "val_text": (
                    str(
                        VAL_TEXT
                    )
                ),

                "report": (
                    str(
                        REPORT_FILE
                    )
                ),
            },
        }


        _save_json(
            REPORT_FILE,
            report,
        )


        # ====================================================
        # Console Summary
        # ====================================================

        print()
        print("=" * 78)

        print(
            "STAGE-1 SMOKE CORPUS COMPLETED"
        )

        print("=" * 78)


        print(
            f"Prepared docs       : "
            f"{len(prepared_documents)}"
        )

        print(
            f"Rejected docs       : "
            f"{cleaning_stats['rejected_documents']}"
        )

        print(
            f"Garbage lines drop  : "
            f"{cleaning_stats['dropped_garbage_lines']}"
        )


        print()
        print(
            "TRAIN"
        )

        print(
            f"  Documents         : "
            f"{train_stats['documents']}"
        )

        print(
            f"  Words             : "
            f"{train_stats['words_total']:,}"
        )

        print(
            f"  Characters        : "
            f"{train_stats['characters_total']:,}"
        )

        print(
            f"  Avg words/doc     : "
            f"{train_stats['words_avg']}"
        )

        print(
            f"  Axis              : "
            f"{train_stats['axis_counts']}"
        )

        print(
            f"  Source            : "
            f"{train_stats['source_counts']}"
        )


        print()
        print(
            "VALIDATION"
        )

        print(
            f"  Documents         : "
            f"{validation_stats['documents']}"
        )

        print(
            f"  Words             : "
            f"{validation_stats['words_total']:,}"
        )

        print(
            f"  Characters        : "
            f"{validation_stats['characters_total']:,}"
        )

        print(
            f"  Avg words/doc     : "
            f"{validation_stats['words_avg']}"
        )

        print(
            f"  Axis              : "
            f"{validation_stats['axis_counts']}"
        )

        print(
            f"  Source            : "
            f"{validation_stats['source_counts']}"
        )


        print()
        print(
            "FILES"
        )

        print(
            f"  Train JSONL       : "
            f"{TRAIN_JSONL}"
        )

        print(
            f"  Val JSONL         : "
            f"{VAL_JSONL}"
        )

        print(
            f"  Train text        : "
            f"{TRAIN_TEXT}"
        )

        print(
            f"  Val text          : "
            f"{VAL_TEXT}"
        )

        print(
            f"  Report            : "
            f"{REPORT_FILE}"
        )


        print()
        print(
            "[PASS] Stage-1 smoke corpus ready."
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "  This Core-100 corpus is only "
            "for smoke-testing the direct "
            "Transformer training pipeline."
        )

        print(
            "  Final Stage-1 training corpus "
            "will later scale to "
            "50K -> 100K -> 300K+ papers."
        )

        print("=" * 78)


    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "STAGE-1 CORPUS PREPARATION FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "Do NOT start tokenizer training "
            "until this step passes."
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()