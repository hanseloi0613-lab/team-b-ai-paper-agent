import argparse
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers import models
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import trainers

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - Corpus-Trained BPE Tokenizer
# Version: team_b_bpe_v1
# ============================================================
#
# GOAL
#
# 외부 pretrained tokenizer를 사용하지 않는다.
#
# 사용하지 않음:
#
#   GPT tokenizer
#   Qwen tokenizer
#   HuggingFace pretrained tokenizer
#
#
# 사용:
#
#   TEAM B scientific corpus
#          ↓
#   BPE vocabulary 직접 학습
#          ↓
#   merge rules 직접 학습
#          ↓
#   tokenizer.json 생성
#
#
# Smoke stage:
#
#   Core-100
#   약 60만 words
#
#
# Final Stage-1:
#
#   50K → 100K → 300K+ papers
#
# corpus가 확장되면 tokenizer도 최종 corpus 기준으로
# 재학습할 수 있다.
#
# ============================================================


TOKENIZER_VERSION = "team_b_bpe_v1"

DEFAULT_VOCAB_SIZE = 8000

MIN_FREQUENCY = 2


# ============================================================
# Special Tokens
#
# IMPORTANT
#
# Stage-1뿐 아니라 앞으로 Stage-2에서 사용할 구조 토큰까지
# 지금 미리 vocabulary에 포함한다.
#
# Transformer를 학습한 뒤 token을 추가하면
# embedding/output head 크기를 변경해야 하기 때문이다.
# ============================================================

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"

TITLE_TOKEN = "<|title|>"
BODY_TOKEN = "<|body|>"

IDEA_TOKEN = "<|idea|>"
EVIDENCE_TOKEN = "<|evidence|>"
CONSTRAINTS_TOKEN = "<|constraints|>"

INTRO_TOKEN = "<|intro|>"
SECTION_BODY_TOKEN = "<|section_body|>"
CONCLUSION_TOKEN = "<|conclusion|>"
REFERENCES_TOKEN = "<|references|>"

SEP_TOKEN = "<|sep|>"


SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    TITLE_TOKEN,
    BODY_TOKEN,
    IDEA_TOKEN,
    EVIDENCE_TOKEN,
    CONSTRAINTS_TOKEN,
    INTRO_TOKEN,
    SECTION_BODY_TOKEN,
    CONCLUSION_TOKEN,
    REFERENCES_TOKEN,
    SEP_TOKEN,
]


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


REPORT_FILE = (
    REPORT_DIR
    / "team_b_bpe_tokenizer_report.json"
)


# ============================================================
# JSON Helper
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

def _validate_inputs() -> None:

    if not TRAIN_TEXT_FILE.exists():
        raise FileNotFoundError(
            "Stage-1 training corpus does not exist.\n"
            f"Expected:\n{TRAIN_TEXT_FILE}\n\n"
            "Run first:\n"
            "uv run python -m app.stage1_smoke_prepare"
        )

    if not VAL_TEXT_FILE.exists():
        raise FileNotFoundError(
            "Stage-1 validation corpus does not exist.\n"
            f"Expected:\n{VAL_TEXT_FILE}"
        )

    train_size = (
        TRAIN_TEXT_FILE.stat().st_size
    )

    val_size = (
        VAL_TEXT_FILE.stat().st_size
    )

    if train_size <= 0:
        raise RuntimeError(
            "Training corpus is empty."
        )

    if val_size <= 0:
        raise RuntimeError(
            "Validation corpus is empty."
        )


# ============================================================
# Build Empty Tokenizer
# ============================================================

def _build_empty_tokenizer() -> Tokenizer:

    # --------------------------------------------------------
    # Empty BPE model.
    #
    # There is NO pretrained vocabulary here.
    # --------------------------------------------------------

    tokenizer = Tokenizer(
        models.BPE(
            unk_token=UNK_TOKEN,
        )
    )

    # --------------------------------------------------------
    # Unicode normalization
    # --------------------------------------------------------

    tokenizer.normalizer = (
        normalizers.NFC()
    )

    # --------------------------------------------------------
    # Byte-level pre-tokenization
    #
    # Advantages:
    #
    # - scientific symbols
    # - punctuation
    # - uncommon strings
    # - identifiers
    #
    # can still be represented without depending heavily
    # on <|unk|>.
    # --------------------------------------------------------

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
        )
    )

    tokenizer.decoder = (
        decoders.ByteLevel()
    )

    return tokenizer


# ============================================================
# Train Tokenizer
# ============================================================

def _train_tokenizer(
    vocab_size: int,
) -> Tokenizer:

    tokenizer = (
        _build_empty_tokenizer()
    )

    trainer = (
        trainers.BpeTrainer(
            vocab_size=vocab_size,

            min_frequency=(
                MIN_FREQUENCY
            ),

            special_tokens=(
                SPECIAL_TOKENS
            ),

            show_progress=True,

            initial_alphabet=(
                pre_tokenizers
                .ByteLevel
                .alphabet()
            ),
        )
    )

    tokenizer.train(
        files=[
            str(
                TRAIN_TEXT_FILE
            )
        ],
        trainer=trainer,
    )

    return tokenizer


# ============================================================
# Special Token Validation
# ============================================================

def _validate_special_tokens(
    tokenizer: Tokenizer,
) -> dict[str, int]:

    token_ids = {}

    for token in SPECIAL_TOKENS:

        token_id = (
            tokenizer.token_to_id(
                token
            )
        )

        if token_id is None:
            raise RuntimeError(
                "Special token missing from tokenizer.\n"
                f"Token: {token}"
            )

        token_ids[
            token
        ] = int(
            token_id
        )

    if (
        len(
            set(
                token_ids.values()
            )
        )
        != len(
            SPECIAL_TOKENS
        )
    ):
        raise RuntimeError(
            "Duplicate special-token IDs detected."
        )

    return token_ids


# ============================================================
# Load Validation Sample
# ============================================================

def _load_validation_sample(
    max_chars: int = 4000,
) -> str:

    text = (
        VAL_TEXT_FILE.read_text(
            encoding="utf-8"
        )
    )

    text = (
        text.strip()
    )

    if not text:
        raise RuntimeError(
            "Validation text is empty."
        )

    return text[
        :max_chars
    ]


# ============================================================
# Stage-2 Synthetic Structural Test
#
# Tokenizer가 향후 Stage-2 input/output 구조도
# 깨지지 않고 encode할 수 있는지 확인한다.
# ============================================================

def _build_stage2_test() -> str:

    return f"""
{BOS_TOKEN}
{TITLE_TOKEN}
Autonomous Onboard Decision-Making for Deep-Space Spacecraft Under Communication Delays

{IDEA_TOKEN}
Investigate whether onboard AI can reduce dependence on real-time ground control while maintaining safety and human oversight.

{EVIDENCE_TOKEN}
Evidence 1: Deep-space communication delays increase the need for onboard autonomy.

{CONSTRAINTS_TOKEN}
Use only supplied evidence for factual claims.
Do not invent numerical values.
Do not invent citations.
Do not exaggerate findings.

{INTRO_TOKEN}
Autonomous decision-making is increasingly important for future deep-space missions.

{SECTION_BODY_TOKEN}
Onboard systems can reduce dependence on continuous ground intervention.

{CONCLUSION_TOKEN}
Reliable autonomy requires both technical capability and appropriate human oversight.

{REFERENCES_TOKEN}
Evidence references are linked to retrieved source documents.

{EOS_TOKEN}
""".strip()


# ============================================================
# Encode Statistics
# ============================================================

def _encoding_stats(
    tokenizer: Tokenizer,
    text: str,
) -> dict:

    encoding = (
        tokenizer.encode(
            text
        )
    )

    ids = (
        encoding.ids
    )

    tokens = (
        encoding.tokens
    )

    if not ids:
        raise RuntimeError(
            "Tokenizer returned zero tokens."
        )

    unk_id = (
        tokenizer.token_to_id(
            UNK_TOKEN
        )
    )

    unknown_count = sum(
        1
        for token_id in ids
        if token_id == unk_id
    )

    char_count = len(
        text
    )

    token_count = len(
        ids
    )

    chars_per_token = (
        char_count
        / token_count
        if token_count
        else 0.0
    )

    return {
        "character_count": (
            char_count
        ),

        "token_count": (
            token_count
        ),

        "unknown_token_count": (
            unknown_count
        ),

        "unknown_token_ratio": (
            round(
                (
                    unknown_count
                    / token_count
                ),
                8,
            )
        ),

        "characters_per_token": (
            round(
                chars_per_token,
                4,
            )
        ),

        "first_100_ids": (
            ids[
                :100
            ]
        ),

        "first_100_tokens": (
            tokens[
                :100
            ]
        ),
    }


# ============================================================
# Round-Trip Test
# ============================================================

def _roundtrip_test(
    tokenizer: Tokenizer,
    text: str,
) -> dict:

    encoding = (
        tokenizer.encode(
            text
        )
    )

    decoded = (
        tokenizer.decode(
            encoding.ids,
            skip_special_tokens=False,
        )
    )

    # ByteLevel decoder may normalize representation slightly,
    # therefore exact string equality is diagnostic rather than
    # a hard failure.

    return {
        "input_preview": (
            text[
                :1000
            ]
        ),

        "decoded_preview": (
            decoded[
                :1000
            ]
        ),

        "exact_match": (
            decoded == text
        ),
    }


# ============================================================
# Tokenizer Vocabulary Diagnostics
# ============================================================

def _vocab_diagnostics(
    tokenizer: Tokenizer,
) -> dict:

    vocab = (
        tokenizer.get_vocab()
    )

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    if vocab_size <= 0:
        raise RuntimeError(
            "Tokenizer vocabulary is empty."
        )

    # Sort by token ID
    ordered_vocab = sorted(
        vocab.items(),
        key=lambda pair: pair[
            1
        ],
    )

    first_tokens = [
        {
            "token": token,
            "id": int(
                token_id
            ),
        }
        for (
            token,
            token_id,
        )
        in ordered_vocab[
            :100
        ]
    ]

    last_tokens = [
        {
            "token": token,
            "id": int(
                token_id
            ),
        }
        for (
            token,
            token_id,
        )
        in ordered_vocab[
            -100:
        ]
    ]

    return {
        "vocab_size": (
            int(
                vocab_size
            )
        ),

        "first_100_vocab_entries": (
            first_tokens
        ),

        "last_100_vocab_entries": (
            last_tokens
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Train TEAM B BPE tokenizer "
                "from our own scientific corpus."
            )
        )
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=(
            "Target BPE vocabulary size. "
            "Default: 8000"
        ),
    )

    args = (
        parser.parse_args()
    )

    vocab_size = int(
        args.vocab_size
    )

    if vocab_size < 1000:
        raise ValueError(
            "vocab_size must be >= 1000."
        )


    print()
    print("=" * 78)

    print(
        "TEAM B - CORPUS-TRAINED "
        "BPE TOKENIZER"
    )

    print("=" * 78)

    print(
        f"Version           : "
        f"{TOKENIZER_VERSION}"
    )

    print(
        f"Training corpus   : "
        f"{TRAIN_TEXT_FILE}"
    )

    print(
        f"Validation corpus : "
        f"{VAL_TEXT_FILE}"
    )

    print(
        f"Target vocab      : "
        f"{vocab_size}"
    )

    print(
        f"Min frequency     : "
        f"{MIN_FREQUENCY}"
    )

    print(
        f"Special tokens    : "
        f"{len(SPECIAL_TOKENS)}"
    )


    try:

        # ====================================================
        # Preflight
        # ====================================================

        _validate_inputs()

        print()
        print(
            "[PRECHECK] Corpus files OK."
        )

        print(
            f"[PRECHECK] Train size : "
            f"{TRAIN_TEXT_FILE.stat().st_size:,} bytes"
        )

        print(
            f"[PRECHECK] Val size   : "
            f"{VAL_TEXT_FILE.stat().st_size:,} bytes"
        )


        # ====================================================
        # Train
        # ====================================================

        print()
        print(
            "[TOKENIZER] Training BPE "
            "from empty vocabulary..."
        )

        tokenizer = (
            _train_tokenizer(
                vocab_size
            )
        )


        # ====================================================
        # Vocabulary
        # ====================================================

        actual_vocab_size = (
            tokenizer.get_vocab_size()
        )

        print()
        print(
            f"[TOKENIZER] Actual vocab size : "
            f"{actual_vocab_size}"
        )


        # ====================================================
        # Special token validation
        # ====================================================

        special_token_ids = (
            _validate_special_tokens(
                tokenizer
            )
        )

        print()
        print(
            "[TOKENIZER] Special token IDs:"
        )

        for (
            token,
            token_id,
        ) in (
            special_token_ids.items()
        ):

            print(
                f"  {token:<22} "
                f"{token_id}"
            )


        # ====================================================
        # Validation sample
        # ====================================================

        validation_sample = (
            _load_validation_sample()
        )

        validation_stats = (
            _encoding_stats(
                tokenizer,
                validation_sample,
            )
        )

        validation_roundtrip = (
            _roundtrip_test(
                tokenizer,
                validation_sample,
            )
        )


        # ====================================================
        # Stage-2 structural test
        # ====================================================

        stage2_sample = (
            _build_stage2_test()
        )

        stage2_stats = (
            _encoding_stats(
                tokenizer,
                stage2_sample,
            )
        )

        stage2_roundtrip = (
            _roundtrip_test(
                tokenizer,
                stage2_sample,
            )
        )


        # ====================================================
        # Vocabulary diagnostics
        # ====================================================

        vocab_diagnostics = (
            _vocab_diagnostics(
                tokenizer
            )
        )


        # ====================================================
        # Save tokenizer
        # ====================================================

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        tokenizer.save(
            str(
                TOKENIZER_FILE
            )
        )

        if not (
            TOKENIZER_FILE.exists()
        ):
            raise RuntimeError(
                "Tokenizer file was not created."
            )


        # ====================================================
        # Reload Test
        #
        # 실제 Transformer에서는 저장된 tokenizer를
        # 다시 load해서 사용할 것이므로 반드시 확인.
        # ====================================================

        reloaded = (
            Tokenizer.from_file(
                str(
                    TOKENIZER_FILE
                )
            )
        )

        reload_vocab_size = (
            reloaded.get_vocab_size()
        )

        if (
            reload_vocab_size
            != actual_vocab_size
        ):
            raise RuntimeError(
                "Tokenizer reload vocab mismatch.\n"
                f"Before save: "
                f"{actual_vocab_size}\n"
                f"After load : "
                f"{reload_vocab_size}"
            )

        reload_test = (
            reloaded.encode(
                "Autonomous spacecraft require "
                "reliable onboard decision-making."
            )
        )

        if not (
            reload_test.ids
        ):
            raise RuntimeError(
                "Reloaded tokenizer failed "
                "to encode text."
            )


        # ====================================================
        # Critical Gate
        #
        # ByteLevel BPE should normally have zero UNK
        # for ordinary UTF-8 scientific English.
        # ====================================================

        if (
            validation_stats[
                "unknown_token_count"
            ]
            > 0
        ):

            print()
            print(
                "[WARN] Validation sample contains "
                f"{validation_stats['unknown_token_count']} "
                "UNK token(s)."
            )


        # ====================================================
        # Report
        # ====================================================

        report = {
            "tokenizer_version": (
                TOKENIZER_VERSION
            ),

            "training_mode": (
                "BPE trained from TEAM B corpus "
                "with no pretrained vocabulary"
            ),

            "requested_vocab_size": (
                vocab_size
            ),

            "actual_vocab_size": (
                actual_vocab_size
            ),

            "minimum_frequency": (
                MIN_FREQUENCY
            ),

            "training_corpus": (
                str(
                    TRAIN_TEXT_FILE
                )
            ),

            "validation_corpus": (
                str(
                    VAL_TEXT_FILE
                )
            ),

            "tokenizer_file": (
                str(
                    TOKENIZER_FILE
                )
            ),

            "special_tokens": (
                SPECIAL_TOKENS
            ),

            "special_token_ids": (
                special_token_ids
            ),

            "validation_sample": {
                "encoding": (
                    validation_stats
                ),

                "roundtrip": (
                    validation_roundtrip
                ),
            },

            "stage2_structure_test": {
                "encoding": (
                    stage2_stats
                ),

                "roundtrip": (
                    stage2_roundtrip
                ),
            },

            "vocabulary_diagnostics": (
                vocab_diagnostics
            ),

            "reload_test": {
                "vocab_size": (
                    reload_vocab_size
                ),

                "encoded_token_count": (
                    len(
                        reload_test.ids
                    )
                ),

                "encoded_ids": (
                    reload_test.ids
                ),

                "encoded_tokens": (
                    reload_test.tokens
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
            "BPE TOKENIZER TRAINING COMPLETED"
        )

        print("=" * 78)

        print(
            f"Requested vocab    : "
            f"{vocab_size}"
        )

        print(
            f"Actual vocab       : "
            f"{actual_vocab_size}"
        )

        print(
            f"Special tokens     : "
            f"{len(SPECIAL_TOKENS)}"
        )


        print()
        print(
            "VALIDATION SAMPLE"
        )

        print(
            f"  Characters       : "
            f"{validation_stats['character_count']:,}"
        )

        print(
            f"  Tokens           : "
            f"{validation_stats['token_count']:,}"
        )

        print(
            f"  Chars/token      : "
            f"{validation_stats['characters_per_token']}"
        )

        print(
            f"  UNK count        : "
            f"{validation_stats['unknown_token_count']}"
        )

        print(
            f"  UNK ratio        : "
            f"{validation_stats['unknown_token_ratio']}"
        )


        print()
        print(
            "STAGE-2 STRUCTURE TEST"
        )

        print(
            f"  Characters       : "
            f"{stage2_stats['character_count']:,}"
        )

        print(
            f"  Tokens           : "
            f"{stage2_stats['token_count']:,}"
        )

        print(
            f"  UNK count        : "
            f"{stage2_stats['unknown_token_count']}"
        )


        print()
        print(
            "RELOAD TEST"
        )

        print(
            f"  Reload vocab     : "
            f"{reload_vocab_size}"
        )

        print(
            f"  Test tokens      : "
            f"{len(reload_test.ids)}"
        )


        print()
        print(
            "FIRST VALIDATION TOKENS"
        )

        print(
            validation_stats[
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
            f"  Tokenizer        : "
            f"{TOKENIZER_FILE}"
        )

        print(
            f"  Report           : "
            f"{REPORT_FILE}"
        )


        print()
        print(
            "[PASS] TEAM B corpus-trained "
            "BPE tokenizer ready."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "  Build tokenized causal-LM dataset "
            "and direct Decoder-only Transformer."
        )

        print("=" * 78)


    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "TOKENIZER TRAINING FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "Do NOT start Transformer training "
            "until tokenizer training passes."
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()