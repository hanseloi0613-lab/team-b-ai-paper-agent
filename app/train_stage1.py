from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from app.transformer_model import (
    DecoderOnlyTransformer,
    TransformerConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_TOKENS_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "stage1_smoke_train_tokens.npy"
)

VAL_TOKENS_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "stage1_smoke_val_tokens.npy"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "data"
    / "checkpoints"
    / "stage1_smoke"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "stage1_smoke_training_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEAM B Stage1 direct Transformer training"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-batches",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokens(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Token file not found: {path}"
        )

    tokens = np.load(
        path,
        mmap_mode="r",
    )

    if tokens.ndim != 1:
        raise ValueError(
            f"Expected 1D token array: {path}"
        )

    return tokens


def get_batch(
    tokens: np.ndarray,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:

    max_start = len(tokens) - seq_len - 1

    if max_start <= 0:
        raise ValueError(
            f"Token array too short for seq_len={seq_len}"
        )

    starts = rng.integers(
        0,
        max_start,
        size=batch_size,
    )

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    for start in starts:
        start = int(start)

        x_np = np.array(
            tokens[start:start + seq_len],
            dtype=np.int64,
            copy=True,
        )

        y_np = np.array(
            tokens[
                start + 1:
                start + seq_len + 1
            ],
            dtype=np.int64,
            copy=True,
        )

        xs.append(
            torch.from_numpy(x_np)
        )

        ys.append(
            torch.from_numpy(y_np)
        )

    x = torch.stack(xs)
    y = torch.stack(ys)

    if device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()

    x = x.to(
        device,
        non_blocking=True,
    )

    y = y.to(
        device,
        non_blocking=True,
    )

    return x, y


@torch.no_grad()
def evaluate(
    model: DecoderOnlyTransformer,
    tokens: np.ndarray,
    *,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    eval_seed: int,
) -> float:
    """
    매 evaluation마다 동일한 seed를 다시 사용한다.

    따라서 initial / step100 / final validation이
    동일한 validation windows에서 측정되어
    loss 감소를 공정하게 비교할 수 있다.
    """

    rng = np.random.default_rng(
        eval_seed
    )

    model.eval()

    losses: list[float] = []

    for _ in range(eval_batches):
        x, y = get_batch(
            tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            rng=rng,
        )

        _, loss = model(
            x,
            y,
        )

        if loss is None:
            raise RuntimeError(
                "Validation loss is None"
            )

        losses.append(
            float(loss.item())
        )

    model.train()

    return float(
        sum(losses) / len(losses)
    )


def effective_warmup_steps(
    requested_warmup: int,
    max_steps: int,
) -> int:
    """
    800-step 본훈련:
      warmup 50 유지

    10-step smoke:
      warmup가 10 step 전체를 먹지 않도록 축소
    """

    short_run_limit = max(
        1,
        max_steps // 10,
    )

    return min(
        requested_warmup,
        short_run_limit,
    )


def learning_rate_for_step(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:

    if step <= warmup_steps:
        return (
            base_lr
            * step
            / max(1, warmup_steps)
        )

    if max_steps <= warmup_steps:
        return base_lr

    progress = (
        step - warmup_steps
    ) / (
        max_steps - warmup_steps
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return (
        min_lr
        + cosine
        * (base_lr - min_lr)
    )


def save_checkpoint(
    *,
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "step":
                step,

            "val_loss":
                val_loss,

            "model_config":
                asdict(model.config),

            "training_args":
                vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print(
        "TEAM B - Stage1 Direct Transformer Training"
    )
    print("=" * 78)

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "Run this inside the WSL team-b-transformer environment."
        )

    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(
        "PyTorch         :",
        torch.__version__,
    )

    print(
        "CUDA runtime    :",
        torch.version.cuda,
    )

    print(
        "GPU             :",
        torch.cuda.get_device_name(0),
    )

    print(
        "Capability      :",
        torch.cuda.get_device_capability(0),
    )

    free_vram, total_vram = (
        torch.cuda.mem_get_info()
    )

    print(
        "VRAM free       :",
        f"{free_vram / 1024**2:.0f} MiB",
    )

    print(
        "VRAM total      :",
        f"{total_vram / 1024**2:.0f} MiB",
    )

    print()

    train_tokens = load_tokens(
        TRAIN_TOKENS_PATH
    )

    val_tokens = load_tokens(
        VAL_TOKENS_PATH
    )

    print(
        "Train tokens    :",
        f"{len(train_tokens):,}",
    )

    print(
        "Val tokens      :",
        f"{len(val_tokens):,}",
    )

    print(
        "Sequence length :",
        args.seq_len,
    )

    print(
        "Micro batch     :",
        args.micro_batch_size,
    )

    print(
        "Grad accum      :",
        args.grad_accum_steps,
    )

    effective_batch = (
        args.micro_batch_size
        * args.grad_accum_steps
    )

    tokens_per_step = (
        args.seq_len
        * effective_batch
    )

    print(
        "Effective batch :",
        effective_batch,
    )

    print(
        "Tokens / step   :",
        tokens_per_step,
    )

    actual_warmup = (
        effective_warmup_steps(
            args.warmup_steps,
            args.max_steps,
        )
    )

    print(
        "Warmup steps    :",
        actual_warmup,
    )

    print()

    config = TransformerConfig(
        vocab_size=8000,
        max_seq_len=args.seq_len,
        d_model=192,
        n_heads=6,
        n_layers=4,
        d_ff=768,
        dropout=0.10,
        bias=True,
        pad_token_id=0,
    )

    model = DecoderOnlyTransformer(
        config
    ).to(device)

    parameter_count = (
        model.parameter_count()
    )

    print(
        "Parameters      :",
        f"{parameter_count:,}",
    )

    print(
        "Random CE approx:",
        f"{math.log(config.vocab_size):.4f}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    train_rng = np.random.default_rng(
        args.seed
    )

    eval_seed = args.seed + 1000

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.cuda.reset_peak_memory_stats()

    print()
    print(
        "Running fixed validation probe..."
    )

    initial_val_loss = evaluate(
        model,
        val_tokens,
        seq_len=args.seq_len,
        batch_size=args.micro_batch_size,
        eval_batches=args.eval_batches,
        device=device,
        eval_seed=eval_seed,
    )

    print(
        "Initial val loss:",
        f"{initial_val_loss:.4f}",
    )

    print(
        "Initial val ppl :",
        f"{math.exp(min(initial_val_loss, 20)):.2f}",
    )

    best_val_loss = initial_val_loss
    best_step = 0

    history: list[dict] = [
        {
            "step": 0,
            "val_loss":
                initial_val_loss,
        }
    ]

    start_time = time.perf_counter()
    log_time = start_time
    tokens_since_log = 0

    model.train()

    print()
    print("=" * 78)
    print("TRAINING START")
    print("=" * 78)

    try:
        for step in range(
            1,
            args.max_steps + 1,
        ):

            lr = learning_rate_for_step(
                step,
                max_steps=args.max_steps,
                warmup_steps=actual_warmup,
                base_lr=args.learning_rate,
                min_lr=args.min_learning_rate,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(
                set_to_none=True
            )

            loss_sum = 0.0

            for _ in range(
                args.grad_accum_steps
            ):
                x, y = get_batch(
                    train_tokens,
                    seq_len=args.seq_len,
                    batch_size=args.micro_batch_size,
                    device=device,
                    rng=train_rng,
                )

                _, loss = model(
                    x,
                    y,
                )

                if loss is None:
                    raise RuntimeError(
                        "Training loss is None"
                    )

                loss_sum += (
                    float(loss.item())
                )

                (
                    loss
                    / args.grad_accum_steps
                ).backward()

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            )

            optimizer.step()

            train_loss = (
                loss_sum
                / args.grad_accum_steps
            )

            tokens_since_log += (
                tokens_per_step
            )

            if (
                step == 1
                or step % args.log_interval == 0
            ):
                now = time.perf_counter()

                elapsed = max(
                    now - log_time,
                    1e-9,
                )

                tokens_per_second = (
                    tokens_since_log
                    / elapsed
                )

                allocated = (
                    torch.cuda.memory_allocated()
                    / 1024**2
                )

                reserved = (
                    torch.cuda.memory_reserved()
                    / 1024**2
                )

                peak = (
                    torch.cuda.max_memory_allocated()
                    / 1024**2
                )

                print(
                    f"step {step:4d}/{args.max_steps}"
                    f" | loss {train_loss:.4f}"
                    f" | lr {lr:.6f}"
                    f" | grad {float(grad_norm):.3f}"
                    f" | {tokens_per_second:,.0f} tok/s"
                    f" | alloc {allocated:.0f} MiB"
                    f" | reserved {reserved:.0f} MiB"
                    f" | peak {peak:.0f} MiB"
                )

                log_time = now
                tokens_since_log = 0

            if (
                step % args.eval_interval == 0
                or step == args.max_steps
            ):
                val_loss = evaluate(
                    model,
                    val_tokens,
                    seq_len=args.seq_len,
                    batch_size=args.micro_batch_size,
                    eval_batches=args.eval_batches,
                    device=device,
                    eval_seed=eval_seed,
                )

                val_ppl = math.exp(
                    min(val_loss, 20)
                )

                print()
                print(
                    f"[EVAL]"
                    f" step={step}"
                    f" val_loss={val_loss:.4f}"
                    f" ppl={val_ppl:.2f}"
                )

                history.append(
                    {
                        "step":
                            step,

                        "train_loss":
                            train_loss,

                        "val_loss":
                            val_loss,

                        "val_perplexity":
                            val_ppl,

                        "learning_rate":
                            lr,
                    }
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step

                    best_path = (
                        CHECKPOINT_DIR
                        / "best.pt"
                    )

                    save_checkpoint(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        val_loss=val_loss,
                        args=args,
                    )

                    print(
                        "[CHECKPOINT] BEST ->",
                        best_path,
                    )

                last_path = (
                    CHECKPOINT_DIR
                    / "last.pt"
                )

                save_checkpoint(
                    path=last_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    val_loss=val_loss,
                    args=args,
                )

                print(
                    "[CHECKPOINT] LAST ->",
                    last_path,
                )

                print()

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print()
            print("=" * 78)
            print("CUDA OOM")
            print("=" * 78)
            print(
                "Fallback command:"
            )
            print(
                "python -m app.train_stage1 "
                "--seq-len 64 "
                "--micro-batch-size 1 "
                "--grad-accum-steps 16"
            )

            torch.cuda.empty_cache()

        raise

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    final_val_loss = evaluate(
        model,
        val_tokens,
        seq_len=args.seq_len,
        batch_size=args.micro_batch_size,
        eval_batches=args.eval_batches,
        device=device,
        eval_seed=eval_seed,
    )

    final_path = (
        CHECKPOINT_DIR
        / "final.pt"
    )

    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        step=args.max_steps,
        val_loss=final_val_loss,
        args=args,
    )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    report = {
        "status":
            "PASS",

        "torch_version":
            torch.__version__,

        "cuda_runtime":
            torch.version.cuda,

        "gpu":
            torch.cuda.get_device_name(0),

        "model_config":
            asdict(config),

        "parameter_count":
            parameter_count,

        "train_token_count":
            int(len(train_tokens)),

        "val_token_count":
            int(len(val_tokens)),

        "max_steps":
            args.max_steps,

        "sequence_length":
            args.seq_len,

        "micro_batch_size":
            args.micro_batch_size,

        "gradient_accumulation_steps":
            args.grad_accum_steps,

        "effective_batch_size":
            effective_batch,

        "tokens_per_optimizer_step":
            tokens_per_step,

        "initial_val_loss":
            initial_val_loss,

        "final_val_loss":
            final_val_loss,

        "best_val_loss":
            best_val_loss,

        "best_step":
            best_step,

        "peak_vram_mib":
            peak_vram,

        "training_seconds":
            total_seconds,

        "history":
            history,

        "final_checkpoint":
            str(final_path),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)

    print(
        "Initial val loss :",
        f"{initial_val_loss:.4f}",
    )

    print(
        "Final val loss   :",
        f"{final_val_loss:.4f}",
    )

    print(
        "Best val loss    :",
        f"{best_val_loss:.4f}",
        f"(step {best_step})",
    )

    print(
        "Peak VRAM        :",
        f"{peak_vram:.1f} MiB",
    )

    print(
        "Training time    :",
        f"{total_seconds:.2f} sec"
        if total_seconds < 60
        else f"{total_seconds / 60:.2f} min",
    )

    print(
        "Final checkpoint :",
        final_path,
    )

    print(
        "Report           :",
        REPORT_PATH,
    )

    print(
        "PIPELINE GATE    : PASS"
    )

    if final_val_loss < initial_val_loss:
        print(
            "LOSS GATE        : PASS"
        )
    else:
        print(
            "LOSS GATE        : NOT YET"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
