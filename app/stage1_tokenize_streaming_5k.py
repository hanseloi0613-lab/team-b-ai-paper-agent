import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "stage1_5k_v1"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = DATA_DIR / "reports"

TOKENIZER_FILE = (
    CACHE_DIR
    / "stage1_5k_bpe_tokenizer.json"
)

TRAIN_JSONL = (
    PROCESSED_DIR
    / "stage1_5k_train.jsonl"
)

VAL_JSONL = (
    PROCESSED_DIR
    / "stage1_5k_val.jsonl"
)

TRAIN_NPY = (
    CACHE_DIR
    / "stage1_5k_train_tokens.npy"
)

VAL_NPY = (
    CACHE_DIR
    / "stage1_5k_val_tokens.npy"
)

REPORT_FILE = (
    REPORT_DIR
    / "stage1_5k_streaming_tokenize_report.json"
)

EXPECTED_TRAIN_DOCS = 4500
EXPECTED_VAL_DOCS = 500

DTYPE = np.uint16

COPY_CHUNK_TOKENS = 5_000_000


def build_document_text(
    row: dict,
    line_no: int,
    label: str,
) -> str:

    title = str(
        row.get("title") or ""
    ).strip()

    body = str(
        row.get("clean_content") or ""
    ).strip()

    if not title:
        raise RuntimeError(
            f"{label}: empty title at line {line_no}"
        )

    if not body:
        raise RuntimeError(
            f"{label}: empty clean_content "
            f"at line {line_no}"
        )

    return (
        "<|bos|>\n"
        "<|title|>\n"
        f"{title}\n"
        "<|body|>\n"
        f"{body}\n"
        "<|eos|>\n\n"
    )


def validate_special_tokens(
    tokenizer: Tokenizer,
) -> dict[str, int]:

    names = [
        "<|bos|>",
        "<|eos|>",
        "<|title|>",
        "<|body|>",
    ]

    ids = {}

    for name in names:
        token_id = tokenizer.token_to_id(name)

        if token_id is None:
            raise RuntimeError(
                f"Missing special token: {name}"
            )

        ids[name] = int(token_id)

    return ids


def smoke_test_one_document(
    tokenizer: Tokenizer,
    special_ids: dict[str, int],
) -> None:

    with TRAIN_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        first_line = None

        for line in f:
            if line.strip():
                first_line = line
                break

    if first_line is None:
        raise RuntimeError(
            "TRAIN JSONL is empty."
        )

    row = json.loads(first_line)

    text = build_document_text(
        row=row,
        line_no=1,
        label="SMOKE",
    )

    encoding = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    ids = np.asarray(
        encoding.ids,
        dtype=DTYPE,
    )

    bos_id = special_ids["<|bos|>"]
    eos_id = special_ids["<|eos|>"]

    bos_count = int(
        np.count_nonzero(
            ids == bos_id
        )
    )

    eos_count = int(
        np.count_nonzero(
            ids == eos_id
        )
    )

    print()
    print("===== ONE-DOCUMENT SMOKE TEST =====")
    print(f"tokens    : {ids.size:,}")
    print(f"BOS count : {bos_count}")
    print(f"EOS count : {eos_count}")

    if bos_count != 1:
        raise RuntimeError(
            f"SMOKE BOS expected 1, got {bos_count}"
        )

    if eos_count != 1:
        raise RuntimeError(
            f"SMOKE EOS expected 1, got {eos_count}"
        )

    print("[PASS] SPECIAL TOKEN SMOKE TEST")


def tokenize_streaming(
    *,
    tokenizer: Tokenizer,
    special_ids: dict[str, int],
    source_jsonl: Path,
    output_npy: Path,
    expected_documents: int,
    label: str,
) -> dict:

    raw_path = output_npy.with_suffix(
        ".tokens.raw.partial"
    )

    if raw_path.exists():
        raw_path.unlink()

    if output_npy.exists():
        output_npy.unlink()

    total_tokens = 0
    document_count = 0

    bos_count = 0
    eos_count = 0

    minimum_id = None
    maximum_id = None

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    bos_id = (
        special_ids["<|bos|>"]
    )

    eos_id = (
        special_ids["<|eos|>"]
    )

    print()
    print("=" * 70)
    print(
        f"{label} STREAMING TOKENIZATION"
    )
    print("=" * 70)
    print(f"source : {source_jsonl}")
    print(f"output : {output_npy}")

    try:
        with (
            source_jsonl.open(
                "r",
                encoding="utf-8",
            ) as fin,
            raw_path.open("wb") as fout,
        ):

            for line_no, line in enumerate(
                fin,
                start=1,
            ):

                if not line.strip():
                    continue

                row = json.loads(line)

                text = build_document_text(
                    row=row,
                    line_no=line_no,
                    label=label,
                )

                encoding = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )

                ids = np.asarray(
                    encoding.ids,
                    dtype=DTYPE,
                )

                if ids.size == 0:
                    raise RuntimeError(
                        f"{label}: zero tokens "
                        f"at line {line_no}"
                    )

                local_min = int(
                    ids.min()
                )

                local_max = int(
                    ids.max()
                )

                if local_min < 0:
                    raise RuntimeError(
                        f"{label}: negative token id"
                    )

                if local_max >= vocab_size:
                    raise RuntimeError(
                        f"{label}: token id "
                        f"{local_max} >= vocab "
                        f"{vocab_size}"
                    )

                ids.tofile(fout)

                total_tokens += int(
                    ids.size
                )

                document_count += 1

                bos_count += int(
                    np.count_nonzero(
                        ids == bos_id
                    )
                )

                eos_count += int(
                    np.count_nonzero(
                        ids == eos_id
                    )
                )

                minimum_id = (
                    local_min
                    if minimum_id is None
                    else min(
                        minimum_id,
                        local_min,
                    )
                )

                maximum_id = (
                    local_max
                    if maximum_id is None
                    else max(
                        maximum_id,
                        local_max,
                    )
                )

                if (
                    document_count % 100 == 0
                    or document_count
                    == expected_documents
                ):
                    print(
                        f"[{label}] "
                        f"{document_count:,}/"
                        f"{expected_documents:,} docs | "
                        f"{total_tokens:,} tokens"
                    )

        if (
            document_count
            != expected_documents
        ):
            raise RuntimeError(
                f"{label}: expected "
                f"{expected_documents} docs, "
                f"got {document_count}"
            )

        if bos_count != expected_documents:
            raise RuntimeError(
                f"{label}: BOS expected "
                f"{expected_documents}, "
                f"got {bos_count}"
            )

        if eos_count != expected_documents:
            raise RuntimeError(
                f"{label}: EOS expected "
                f"{expected_documents}, "
                f"got {eos_count}"
            )

        expected_bytes = (
            total_tokens
            * np.dtype(DTYPE).itemsize
        )

        actual_bytes = raw_path.stat().st_size

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"{label}: raw byte mismatch: "
                f"expected={expected_bytes}, "
                f"actual={actual_bytes}"
            )

        print()
        print(
            f"[{label}] raw token stream complete"
        )
        print(
            f"[{label}] creating .npy "
            f"without loading all tokens into RAM..."
        )

        source = np.memmap(
            raw_path,
            dtype=DTYPE,
            mode="r",
            shape=(total_tokens,),
        )

        target = np.lib.format.open_memmap(
            output_npy,
            mode="w+",
            dtype=DTYPE,
            shape=(total_tokens,),
        )

        for start in range(
            0,
            total_tokens,
            COPY_CHUNK_TOKENS,
        ):
            end = min(
                start + COPY_CHUNK_TOKENS,
                total_tokens,
            )

            target[start:end] = (
                source[start:end]
            )

        target.flush()

        del target
        del source

        check = np.load(
            output_npy,
            mmap_mode="r",
        )

        if check.shape != (
            total_tokens,
        ):
            raise RuntimeError(
                f"{label}: final shape mismatch "
                f"{check.shape}"
            )

        if check.dtype != DTYPE:
            raise RuntimeError(
                f"{label}: final dtype mismatch "
                f"{check.dtype}"
            )

        final_min = int(
            check.min()
        )

        final_max = int(
            check.max()
        )

        if (
            final_min != minimum_id
            or final_max != maximum_id
        ):
            raise RuntimeError(
                f"{label}: final token range "
                f"mismatch"
            )

        del check

        raw_path.unlink()

        print()
        print(
            f"[PASS] {label}"
        )
        print(
            f"  documents : "
            f"{document_count:,}"
        )
        print(
            f"  tokens    : "
            f"{total_tokens:,}"
        )
        print(
            f"  BOS       : "
            f"{bos_count:,}"
        )
        print(
            f"  EOS       : "
            f"{eos_count:,}"
        )
        print(
            f"  ID range  : "
            f"{minimum_id} .. "
            f"{maximum_id}"
        )
        print(
            f"  dtype     : "
            f"{np.dtype(DTYPE)}"
        )
        print(
            f"  file      : "
            f"{output_npy}"
        )

        return {
            "documents": document_count,
            "tokens": total_tokens,
            "bos_count": bos_count,
            "eos_count": eos_count,
            "min_token_id": minimum_id,
            "max_token_id": maximum_id,
            "dtype": str(
                np.dtype(DTYPE)
            ),
            "output_file": str(
                output_npy
            ),
        }

    except Exception:
        if raw_path.exists():
            raw_path.unlink()

        raise


def main() -> None:

    print("=" * 70)
    print(
        "TEAM B - STAGE1 5K "
        "STREAMING TOKENIZER"
    )
    print("=" * 70)

    for path in (
        TOKENIZER_FILE,
        TRAIN_JSONL,
        VAL_JSONL,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    print(
        f"Tokenizer vocab : "
        f"{vocab_size:,}"
    )

    if vocab_size != 8000:
        raise RuntimeError(
            f"Expected vocab=8000, "
            f"got {vocab_size}"
        )

    if (
        vocab_size - 1
        > np.iinfo(DTYPE).max
    ):
        raise RuntimeError(
            "Vocabulary does not fit uint16."
        )

    special_ids = (
        validate_special_tokens(
            tokenizer
        )
    )

    print()
    print("Special token IDs")

    for name, token_id in (
        special_ids.items()
    ):
        print(
            f"  {name:12s} "
            f"-> {token_id}"
        )

    smoke_test_one_document(
        tokenizer,
        special_ids,
    )

    train_stats = tokenize_streaming(
        tokenizer=tokenizer,
        special_ids=special_ids,
        source_jsonl=TRAIN_JSONL,
        output_npy=TRAIN_NPY,
        expected_documents=(
            EXPECTED_TRAIN_DOCS
        ),
        label="TRAIN",
    )

    val_stats = tokenize_streaming(
        tokenizer=tokenizer,
        special_ids=special_ids,
        source_jsonl=VAL_JSONL,
        output_npy=VAL_NPY,
        expected_documents=(
            EXPECTED_VAL_DOCS
        ),
        label="VAL",
    )

    report = {
        "tokenizer": str(
            TOKENIZER_FILE
        ),
        "vocab_size": vocab_size,
        "dtype": str(
            np.dtype(DTYPE)
        ),
        "special_token_ids": (
            special_ids
        ),
        "train": train_stats,
        "validation": val_stats,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "FINAL STREAMING TOKENIZATION PASS"
    )
    print("=" * 70)
    print(
        f"TRAIN tokens : "
        f"{train_stats['tokens']:,}"
    )
    print(
        f"VAL tokens   : "
        f"{val_stats['tokens']:,}"
    )
    print(
        f"Report       : "
        f"{REPORT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
