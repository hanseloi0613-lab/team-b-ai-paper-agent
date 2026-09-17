from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """
    TEAM B 직접 구현 Decoder-only Transformer 설정.

    GTX 1050 2GB smoke training 기본값:
      vocab_size   = 8000
      max_seq_len  = 128
      d_model      = 192
      n_heads      = 6
      n_layers     = 4
      d_ff         = 768

    외부 pretrained model / nn.Transformer / nn.MultiheadAttention 미사용.
    """

    vocab_size: int = 8000
    max_seq_len: int = 128

    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 768

    dropout: float = 0.10
    bias: bool = True

    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model({self.d_model}) must be divisible by "
                f"n_heads({self.n_heads})"
            )

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")

        if self.max_seq_len <= 1:
            raise ValueError("max_seq_len must be > 1")


class CausalSelfAttention(nn.Module):
    """
    직접 구현 Multi-Head Causal Self-Attention.

    핵심:
      Q = XWq
      K = XWk
      V = XWv

      Attention(Q, K, V)
        = softmax(QK^T / sqrt(d_head) + causal_mask)V

    PyTorch nn.MultiheadAttention을 사용하지 않는다.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V를 한 번의 projection으로 계산한 뒤 3개로 분리.
        self.qkv_projection = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=config.bias,
        )

        self.output_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # 미래 token을 볼 수 없도록 lower triangular mask.
        causal_mask = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len,
                dtype=torch.bool,
            )
        )

        causal_mask = causal_mask.view(
            1,
            1,
            config.max_seq_len,
            config.max_seq_len,
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_projection(x)

        query, key, value = qkv.chunk(3, dim=-1)

        # [B, T, C]
        # ->
        # [B, H, T, D_head]
        query = query.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # [B, H, T, D]
        # @
        # [B, H, D, T]
        # ->
        # [B, H, T, T]
        attention_scores = (
            query @ key.transpose(-2, -1)
        ) * self.scale

        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        # [B, H, T, T] @ [B, H, T, D]
        # ->
        # [B, H, T, D]
        output = attention_weights @ value

        # [B, H, T, D]
        # ->
        # [B, T, H, D]
        # ->
        # [B, T, C]
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )

        output = self.output_projection(output)
        output = self.residual_dropout(output)

        return output


class FeedForward(nn.Module):
    """
    Transformer MLP.

    Linear
      -> GELU
      -> Linear
      -> Dropout
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.fc_in = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=config.bias,
        )

        self.fc_out = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.fc_out(x)
        x = self.dropout(x)

        return x


class DecoderBlock(nn.Module):
    """
    Pre-LayerNorm Decoder Block.

    x = x + Attention(LN(x))
    x = x + FFN(LN(x))
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.ln_attention = nn.LayerNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.ln_ffn = nn.LayerNorm(
            config.d_model
        )

        self.feed_forward = FeedForward(
            config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(
            self.ln_attention(x)
        )

        x = x + self.feed_forward(
            self.ln_ffn(x)
        )

        return x


class DecoderOnlyTransformer(nn.Module):
    """
    TEAM B 직접 구현 Decoder-only Transformer Language Model.

    구조:

      token ids
          ↓
      Token Embedding
          +
      Position Embedding
          ↓
      Decoder Block × N
          ↓
      Final LayerNorm
          ↓
      LM Head
          ↓
      next-token logits
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # 먼저 각각의 weight를 초기화.
        self.apply(self._init_weights)

        # Input embedding과 output projection weight 공유.
        # GPT 계열에서 흔히 사용하는 weight tying.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, seq_len]"
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(
            0,
            seq_len,
            dtype=torch.long,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets must have same shape as input_ids"
                )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.config.vocab_size,
                ),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        기본 autoregressive generation.

        Smoke 단계용.
        긴 최종 논문 생성은 Stage2에서
        section-by-section generation으로 확장 예정.
        """

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        self.eval()

        generated = input_ids

        for _ in range(max_new_tokens):

            context = generated[
                :,
                -self.config.max_seq_len:
            ]

            logits, _ = self(
                context
            )

            next_token_logits = (
                logits[:, -1, :] / temperature
            )

            if top_k is not None:
                k = min(
                    top_k,
                    next_token_logits.size(-1),
                )

                values, _ = torch.topk(
                    next_token_logits,
                    k,
                )

                threshold = values[:, [-1]]

                next_token_logits = (
                    next_token_logits.masked_fill(
                        next_token_logits < threshold,
                        float("-inf"),
                    )
                )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if eos_token_id is not None:
                if torch.all(
                    next_token.squeeze(-1)
                    == eos_token_id
                ):
                    break

        return generated

    def parameter_count(
        self,
        trainable_only: bool = True,
    ) -> int:
        parameters = self.parameters()

        if trainable_only:
            return sum(
                p.numel()
                for p in parameters
                if p.requires_grad
            )

        return sum(
            p.numel()
            for p in parameters
        )

    def config_dict(self) -> dict:
        return asdict(self.config)


def run_smoke_test() -> None:
    """
    모델 구조 자체를 검증하는 최소 테스트.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = TransformerConfig()

    model = DecoderOnlyTransformer(
        config
    ).to(device)

    print("=" * 70)
    print("TEAM B Transformer Model Smoke Test")
    print("=" * 70)

    print("Device          :", device)

    if device.type == "cuda":
        print(
            "GPU             :",
            torch.cuda.get_device_name(0),
        )

    print(
        "Parameters      :",
        f"{model.parameter_count():,}",
    )

    print(
        "Config          :",
        model.config_dict(),
    )

    batch_size = 1
    seq_len = 32

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    logits, loss = model(
        input_ids,
        targets,
    )

    print(
        "Input shape     :",
        tuple(input_ids.shape),
    )

    print(
        "Logits shape    :",
        tuple(logits.shape),
    )

    print(
        "Initial loss    :",
        float(loss.item()),
    )

    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

        print(
            "Peak VRAM       :",
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB",
        )

    expected_random_loss = math.log(
        config.vocab_size
    )

    print(
        "ln(vocab_size)  :",
        f"{expected_random_loss:.4f}",
    )

    print("Forward         : PASS")
    print("Backward        : PASS")
    print("MODEL GATE      : PASS")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()