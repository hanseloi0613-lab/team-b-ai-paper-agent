import json
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - Stage-1 Causal LM Corpus Tokenization
# Version: stage1_tokenize_corpus_v1
# ============================================================
#
# INPUT
#
#   stage1_smoke_train.txt
#   stage1_smoke_val.txt
#
#   team_b_bpe_tokenizer.json
#
#
# OUTPUT
#
#   stage1_smoke_train_tokens.npy
#   stage1_smoke_val_tokens.npy
#
#
# PURPOSE
#
# Scientific text
#       ↓
# TEAM B BPE tokenizer
#       ↓
# token IDs
#       ↓
# contiguous causal LM stream
#
#
# Later training:
#
#   tokens:
#
#   [10, 25, 91, 4, 581, ...]
#
#   input:
#
#   [10, 25, 91, 4]
#
#   target:
#
#   [25, 91, 4, 581]
#
#
# Transformer learns:
#
#   P(next token | previous tokens)
#
#
# IMPORTANT
#
# This is still the Core-100 SMOKE dataset.
#
# It validates the entire direct-Transformer pipeline.
#
# Final Stage-1 corpus later scales to:
#
#   50K
#   100K
#   300K+
#
# ============================================================


PIPELINE_VERSION = (
    "stage1_tokenize_corpus_v1"
)


# ============================================================
# Expected tokenizer
# ============================================================

EXPECTED_VOCAB_SIZE = 8000


# ============================================================
# Paths
# ============================================================

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


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


# ============================================================
# Input files
# ============================================================

TRAIN_TEXT_FILE = (
    PROCESSED_DIR
    / "stage1_smoke_train.txt"
)


VAL_TEXT_FILE = (
    PROCESSED_DIR
    / "stage1_smoke_val.txt"
)


TOKENIZER_FILE = (
    CACHE_DIR
    / "team_b_bpe_tokenizer.json"
)


# ============================================================
# Output files
# ============================================================

TRAIN_TOKENS_FILE = (
    CACHE_DIR
    / "stage1_smoke_train_tokens.npy"
)


VAL_TOKENS_FILE = (
    CACHE_DIR
    / "stage1_smoke_val_tokens.npy"
)


REPORT_FILE = (
    REPORT_DIR
    / "stage1_tokenize_corpus_report.json"
)


# ============================================================
# Structural Tokens
# ============================================================

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|title|>",
    "<|body|>",
    "<|idea|>",
    "<|evidence|>",
    "<|constraints|>",
    "<|intro|>",
    "<|section_body|>",
    "<|conclusion|>",
    "<|references|>",
    "<|sep|>",
]


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
# Preflight
# ============================================================

def _validate_input_files() -> None:

    required = [
        TRAIN_TEXT_FILE,
        VAL_TEXT_FILE,
        TOKENIZER_FILE,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Required Stage-1 artifact(s) missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


# ============================================================
# Load Tokenizer
# ============================================================

def _load_tokenizer() -> Tokenizer:

    tokenizer = (
        Tokenizer.from_file(
            str(
                TOKENIZER_FILE
            )
        )
    )

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    if (
        vocab_size
        != EXPECTED_VOCAB_SIZE
    ):

        raise RuntimeError(
            "Tokenizer vocabulary mismatch.\n"
            f"Expected: "
            f"{EXPECTED_VOCAB_SIZE}\n"
            f"Actual  : "
            f"{vocab_size}"
        )

    return tokenizer


# ============================================================
# Special Token QA
# ============================================================

def _validate_special_tokens(
    tokenizer: Tokenizer,
) -> dict[str, int]:

    result = {}

    for token in SPECIAL_TOKENS:

        token_id = (
            tokenizer.token_to_id(
                token
            )
        )

        if token_id is None:

            raise RuntimeError(
                "Missing special token.\n"
                f"Token: {token}"
            )

        result[
            token
        ] = int(
            token_id
        )

    if (
        len(
            set(
                result.values()
            )
        )
        != len(
            result
        )
    ):

        raise RuntimeError(
            "Special-token ID collision detected."
        )

    return result


# ============================================================
# Load Text
# ============================================================

def _load_text(
    path: Path,
) -> str:

    text = (
        path.read_text(
            encoding="utf-8"
        )
    )

    if not (
        text.strip()
    ):

        raise RuntimeError(
            f"Corpus text is empty: {path}"
        )

    return text


# ============================================================
# Tokenize
# ============================================================

def _tokenize_text(
    *,
    tokenizer: Tokenizer,
    text: str,
    label: str,
) -> np.ndarray:

    print()
    print(
        f"[TOKENIZE] Encoding {label} corpus..."
    )

    encoding = (
        tokenizer.encode(
            text,
            add_special_tokens=True,
        )
    )

    token_ids = (
        encoding.ids
    )

    if not token_ids:

        raise RuntimeError(
            f"{label}: tokenizer produced zero tokens."
        )

    max_token_id = max(
        token_ids
    )

    min_token_id = min(
        token_ids
    )

    if (
        min_token_id
        < 0
    ):

        raise RuntimeError(
            f"{label}: negative token ID detected."
        )

    if (
        max_token_id
        >= EXPECTED_VOCAB_SIZE
    ):

        raise RuntimeError(
            f"{label}: token ID exceeds vocabulary.\n"
            f"Maximum ID : "
            f"{max_token_id}\n"
            f"Vocab size : "
            f"{EXPECTED_VOCAB_SIZE}"
        )

    # ========================================================
    # vocab=8000
    #
    # uint16 range:
    #
    # 0 ~ 65535
    #
    # therefore uint16 is sufficient and compact.
    # ========================================================

    tokens = np.asarray(
        token_ids,
        dtype=np.uint16,
    )

    print(
        f"[TOKENIZE] {label} tokens : "
        f"{tokens.size:,}"
    )

    print(
        f"[TOKENIZE] Min token ID : "
        f"{int(tokens.min())}"
    )

    print(
        f"[TOKENIZE] Max token ID : "
        f"{int(tokens.max())}"
    )

    return tokens


# ============================================================
# Token Statistics
# ============================================================

def _token_stats(
    *,
    tokens: np.ndarray,
    source_text: str,
    tokenizer: Tokenizer,
    special_token_ids: dict[str, int],
) -> dict:

    token_count = int(
        tokens.size
    )

    char_count = len(
        source_text
    )

    word_count = len(
        source_text.split()
    )

    unknown_id = (
        special_token_ids[
            "<|unk|>"
        ]
    )

    bos_id = (
        special_token_ids[
            "<|bos|>"
        ]
    )

    eos_id = (
        special_token_ids[
            "<|eos|>"
        ]
    )

    title_id = (
        special_token_ids[
            "<|title|>"
        ]
    )

    body_id = (
        special_token_ids[
            "<|body|>"
        ]
    )


    # ========================================================
    # Counts
    # ========================================================

    unknown_count = int(
        np.count_nonzero(
            tokens
            == unknown_id
        )
    )

    bos_count = int(
        np.count_nonzero(
            tokens
            == bos_id
        )
    )

    eos_count = int(
        np.count_nonzero(
            tokens
            == eos_id
        )
    )

    title_count = int(
        np.count_nonzero(
            tokens
            == title_id
        )
    )

    body_count = int(
        np.count_nonzero(
            tokens
            == body_id
        )
    )


    # ========================================================
    # Ratios
    # ========================================================

    chars_per_token = (
        char_count
        / token_count
    )

    words_per_token = (
        word_count
        / token_count
    )

    tokens_per_word = (
        token_count
        / word_count
        if word_count
        else 0.0
    )


    # ========================================================
    # First IDs / Tokens
    # ========================================================

    first_ids = [
        int(
            value
        )
        for value in tokens[
            :100
        ]
    ]

    first_tokens = [
        tokenizer.id_to_token(
            token_id
        )
        for token_id in first_ids
    ]


    return {
        "characters": (
            int(
                char_count
            )
        ),

        "words": (
            int(
                word_count
            )
        ),

        "tokens": (
            token_count
        ),

        "characters_per_token": (
            round(
                float(
                    chars_per_token
                ),
                4,
            )
        ),

        "words_per_token": (
            round(
                float(
                    words_per_token
                ),
                6,
            )
        ),

        "tokens_per_word": (
            round(
                float(
                    tokens_per_word
                ),
                6,
            )
        ),

        "unknown_tokens": (
            unknown_count
        ),

        "unknown_ratio": (
            round(
                float(
                    unknown_count
                    / token_count
                ),
                10,
            )
        ),

        "bos_count": (
            bos_count
        ),

        "eos_count": (
            eos_count
        ),

        "title_count": (
            title_count
        ),

        "body_count": (
            body_count
        ),

        "minimum_token_id": (
            int(
                tokens.min()
            )
        ),

        "maximum_token_id": (
            int(
                tokens.max()
            )
        ),

        "first_100_ids": (
            first_ids
        ),

        "first_100_tokens": (
            first_tokens
        ),
    }


# ============================================================
# Structural QA
# ============================================================

def _validate_structure(
    *,
    stats: dict,
    label: str,
) -> None:

    # ========================================================
    # Every LM document prepared earlier should contain:
    #
    # <|bos|>
    # <|title|>
    # <|body|>
    # <|eos|>
    #
    # Counts should therefore match.
    # ========================================================

    structural_counts = {
        "bos": (
            stats[
                "bos_count"
            ]
        ),

        "eos": (
            stats[
                "eos_count"
            ]
        ),

        "title": (
            stats[
                "title_count"
            ]
        ),

        "body": (
            stats[
                "body_count"
            ]
        ),
    }

    unique_counts = set(
        structural_counts.values()
    )

    if (
        len(
            unique_counts
        )
        != 1
    ):

        raise RuntimeError(
            f"{label}: structural-token count mismatch.\n"
            f"{structural_counts}"
        )

    document_count = (
        stats[
            "bos_count"
        ]
    )

    if (
        document_count
        <= 0
    ):

        raise RuntimeError(
            f"{label}: no BOS/EOS documents detected."
        )


# ============================================================
# Save Array
# ============================================================

def _save_tokens(
    path: Path,
    tokens: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        tokens,
        allow_pickle=False,
    )

    if not path.exists():

        raise RuntimeError(
            f"Token file was not created: {path}"
        )


# ============================================================
# Reload Verification
# ============================================================

def _verify_saved_array(
    *,
    path: Path,
    expected_tokens: np.ndarray,
) -> dict:

    loaded = (
        np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )
    )

    if (
        loaded.shape
        != expected_tokens.shape
    ):

        raise RuntimeError(
            "Saved token-array shape mismatch.\n"
            f"File     : {path}\n"
            f"Expected : {expected_tokens.shape}\n"
            f"Actual   : {loaded.shape}"
        )

    if (
        loaded.dtype
        != np.uint16
    ):

        raise RuntimeError(
            "Saved token-array dtype mismatch.\n"
            f"Expected : uint16\n"
            f"Actual   : {loaded.dtype}"
        )

    # ========================================================
    # Spot-check beginning / middle / end
    # ========================================================

    size = int(
        loaded.size
    )

    sample_indices = sorted(
        set(
            [
                0,
                min(
                    10,
                    size - 1,
                ),
                size // 2,
                max(
                    0,
                    size - 11,
                ),
                size - 1,
            ]
        )
    )

    mismatches = []

    for index in sample_indices:

        if (
            int(
                loaded[
                    index
                ]
            )
            != int(
                expected_tokens[
                    index
                ]
            )
        ):

            mismatches.append(
                int(
                    index
                )
            )

    if mismatches:

        raise RuntimeError(
            "Saved token-array verification failed.\n"
            f"Mismatch indexes: {mismatches}"
        )

    return {
        "shape": (
            list(
                loaded.shape
            )
        ),

        "dtype": (
            str(
                loaded.dtype
            )
        ),

        "size_bytes": (
            int(
                path.stat().st_size
            )
        ),

        "spot_check_indices": (
            sample_indices
        ),
    }


# ============================================================
# Calculate Number of LM Training Windows
# ============================================================

def _window_counts(
    *,
    token_count: int,
) -> dict:

    block_sizes = [
        128,
        256,
        512,
        1024,
    ]

    counts = {}

    for block_size in block_sizes:

        # input requires block_size tokens
        # target requires the following shifted token.
        usable = max(
            0,
            token_count
            - 1
        )

        non_overlapping = (
            usable
            // block_size
        )

        counts[
            str(
                block_size
            )
        ] = {
            "block_size": (
                block_size
            ),

            "non_overlapping_sequences": (
                int(
                    non_overlapping
                )
            ),
        }

    return counts


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - Stage-1 "
        "Causal LM Corpus Tokenization"
    )

    print("=" * 78)

    print(
        f"Version            : "
        f"{PIPELINE_VERSION}"
    )

    print(
        f"Expected vocab     : "
        f"{EXPECTED_VOCAB_SIZE}"
    )

    print(
        f"Tokenizer          : "
        f"{TOKENIZER_FILE}"
    )


    try:

        # ====================================================
        # Preflight
        # ====================================================

        _validate_input_files()

        print()
        print(
            "[PRECHECK] Required files found."
        )


        # ====================================================
        # Tokenizer
        # ====================================================

        tokenizer = (
            _load_tokenizer()
        )

        print(
            f"[TOKENIZER] Vocabulary : "
            f"{tokenizer.get_vocab_size()}"
        )

        special_token_ids = (
            _validate_special_tokens(
                tokenizer
            )
        )

        print(
            "[TOKENIZER] Special tokens PASS."
        )


        # ====================================================
        # Load Corpus
        # ====================================================

        print()
        print(
            "[LOAD] Reading Stage-1 text..."
        )

        train_text = (
            _load_text(
                TRAIN_TEXT_FILE
            )
        )

        val_text = (
            _load_text(
                VAL_TEXT_FILE
            )
        )

        print(
            f"[LOAD] Train chars : "
            f"{len(train_text):,}"
        )

        print(
            f"[LOAD] Val chars   : "
            f"{len(val_text):,}"
        )


        # ====================================================
        # Tokenize
        # ====================================================

        train_tokens = (
            _tokenize_text(
                tokenizer=tokenizer,
                text=train_text,
                label="TRAIN",
            )
        )

        val_tokens = (
            _tokenize_text(
                tokenizer=tokenizer,
                text=val_text,
                label="VALIDATION",
            )
        )


        # ====================================================
        # Stats
        # ====================================================

        train_stats = (
            _token_stats(
                tokens=train_tokens,
                source_text=train_text,
                tokenizer=tokenizer,
                special_token_ids=(
                    special_token_ids
                ),
            )
        )

        val_stats = (
            _token_stats(
                tokens=val_tokens,
                source_text=val_text,
                tokenizer=tokenizer,
                special_token_ids=(
                    special_token_ids
                ),
            )
        )


        # ====================================================
        # Structural QA
        # ====================================================

        _validate_structure(
            stats=train_stats,
            label="TRAIN",
        )

        _validate_structure(
            stats=val_stats,
            label="VALIDATION",
        )

        print()
        print(
            "[QA] Structural tokens PASS."
        )


        # ====================================================
        # UNK Gate
        # ====================================================

        if (
            train_stats[
                "unknown_tokens"
            ]
            != 0
        ):

            raise RuntimeError(
                "TRAIN corpus contains UNK tokens.\n"
                f"UNK count: "
                f"{train_stats['unknown_tokens']}"
            )

        if (
            val_stats[
                "unknown_tokens"
            ]
            != 0
        ):

            raise RuntimeError(
                "VALIDATION corpus contains UNK tokens.\n"
                f"UNK count: "
                f"{val_stats['unknown_tokens']}"
            )

        print(
            "[QA] UNK tokens = 0."
        )


        # ====================================================
        # Save
        # ====================================================

        print()
        print(
            "[SAVE] Writing token arrays..."
        )

        _save_tokens(
            TRAIN_TOKENS_FILE,
            train_tokens,
        )

        _save_tokens(
            VAL_TOKENS_FILE,
            val_tokens,
        )


        # ====================================================
        # Reload QA
        # ====================================================

        train_reload = (
            _verify_saved_array(
                path=TRAIN_TOKENS_FILE,
                expected_tokens=train_tokens,
            )
        )

        val_reload = (
            _verify_saved_array(
                path=VAL_TOKENS_FILE,
                expected_tokens=val_tokens,
            )
        )

        print(
            "[SAVE] Reload verification PASS."
        )


        # ====================================================
        # Window estimates
        # ====================================================

        train_windows = (
            _window_counts(
                token_count=(
                    train_stats[
                        "tokens"
                    ]
                )
            )
        )

        val_windows = (
            _window_counts(
                token_count=(
                    val_stats[
                        "tokens"
                    ]
                )
            )
        )


        # ====================================================
        # Report
        # ====================================================

        report = {
            "pipeline_version": (
                PIPELINE_VERSION
            ),

            "tokenizer": {
                "file": (
                    str(
                        TOKENIZER_FILE
                    )
                ),

                "vocab_size": (
                    tokenizer
                    .get_vocab_size()
                ),

                "special_token_ids": (
                    special_token_ids
                ),
            },

            "train": {
                "text_file": (
                    str(
                        TRAIN_TEXT_FILE
                    )
                ),

                "tokens_file": (
                    str(
                        TRAIN_TOKENS_FILE
                    )
                ),

                "stats": (
                    train_stats
                ),

                "saved_array": (
                    train_reload
                ),

                "sequence_windows": (
                    train_windows
                ),
            },

            "validation": {
                "text_file": (
                    str(
                        VAL_TEXT_FILE
                    )
                ),

                "tokens_file": (
                    str(
                        VAL_TOKENS_FILE
                    )
                ),

                "stats": (
                    val_stats
                ),

                "saved_array": (
                    val_reload
                ),

                "sequence_windows": (
                    val_windows
                ),
            },
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
            "STAGE-1 TOKENIZATION COMPLETED"
        )

        print("=" * 78)


        print(
            "TRAIN"
        )

        print(
            f"  Characters       : "
            f"{train_stats['characters']:,}"
        )

        print(
            f"  Words            : "
            f"{train_stats['words']:,}"
        )

        print(
            f"  Tokens           : "
            f"{train_stats['tokens']:,}"
        )

        print(
            f"  Chars/token      : "
            f"{train_stats['characters_per_token']}"
        )

        print(
            f"  Tokens/word      : "
            f"{train_stats['tokens_per_word']}"
        )

        print(
            f"  Documents (BOS)  : "
            f"{train_stats['bos_count']}"
        )

        print(
            f"  UNK              : "
            f"{train_stats['unknown_tokens']}"
        )


        print()
        print(
            "VALIDATION"
        )

        print(
            f"  Characters       : "
            f"{val_stats['characters']:,}"
        )

        print(
            f"  Words            : "
            f"{val_stats['words']:,}"
        )

        print(
            f"  Tokens           : "
            f"{val_stats['tokens']:,}"
        )

        print(
            f"  Chars/token      : "
            f"{val_stats['characters_per_token']}"
        )

        print(
            f"  Tokens/word      : "
            f"{val_stats['tokens_per_word']}"
        )

        print(
            f"  Documents (BOS)  : "
            f"{val_stats['bos_count']}"
        )

        print(
            f"  UNK              : "
            f"{val_stats['unknown_tokens']}"
        )


        print()
        print(
            "TRAIN WINDOW ESTIMATES"
        )

        for (
            block_size,
            item,
        ) in (
            train_windows.items()
        ):

            print(
                f"  seq={block_size:<4} "
                f"→ "
                f"{item['non_overlapping_sequences']:,} "
                f"sequences"
            )


        print()
        print(
            "FIRST TRAIN TOKENS"
        )

        print(
            train_stats[
                "first_100_tokens"
            ][
                :50
            ]
        )


        print()
        print(
            "FILES"
        )

        print(
            f"  Train tokens     : "
            f"{TRAIN_TOKENS_FILE}"
        )

        print(
            f"  Val tokens       : "
            f"{VAL_TOKENS_FILE}"
        )

        print(
            f"  Report           : "
            f"{REPORT_FILE}"
        )


        print()
        print(
            "[PASS] Stage-1 causal-LM "
            "token dataset ready."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "  Implement direct Decoder-only "
            "Transformer and CUDA training."
        )

        print("=" * 78)


    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "STAGE-1 TOKENIZATION FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "Do NOT start Transformer training "
            "until tokenization passes."
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()