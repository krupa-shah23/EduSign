"""
models/transformer.py
----------------------
The C2 backbone: a pure transformer encoder over landmark sequences.

No convolution anywhere in this file — that's the explicit scoping clause
that keeps the architecture novel (see Month 1 plan, "why transformer not
CNN/GCN"). Input is the ~283-dim per-frame landmark vector defined in
configs/landmark_config.py, not raw pixels.

Components (each independently testable):
    LandmarkEmbedding   — projects raw per-frame features into hidden_dim
    PositionalEncoding   — fixed sinusoidal position signal added post-projection
    TransformerEncoder   — stack of nn.TransformerEncoderLayer (this IS the backbone)
    MaskedMeanPooling    — collapses (B, T, D) -> (B, D) respecting padding mask
    ClassifierHead       — final linear layer -> n_classes logits (WLASL ISLR task)
    SignEncoder          — wires all of the above together, exposes forward()

Design note: each piece is its own nn.Module (not just inline ops in one
forward()) so they can be unit-tested in isolation and so D can later
import LandmarkEmbedding / TransformerEncoder separately when attaching
the C3 gating head, without needing the classifier head at all.

Everything in this file is runnable with dummy/random tensors — no dataset
required. See `if __name__ == "__main__"` at the bottom for a smoke test
that exercises every shape transition with fake data.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ============================================================================
# 1. LandmarkEmbedding
# ============================================================================

class LandmarkEmbedding(nn.Module):
    """
    Projects the raw per-frame landmark vector into the model's hidden_dim.

    Input  : (B, T, input_dim)   e.g. input_dim = 283 from landmark_config.feature_dim()
    Output : (B, T, hidden_dim)

    Why a plain Linear and not something fancier: the per-frame feature is
    already a structured, normalized vector (see extract_landmarks.py) —
    there's no spatial locality to exploit the way there would be for raw
    pixels, so a single learned linear projection is the right amount of
    capacity here. Keep this simple; let the transformer layers do the work.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) float tensor — raw/normalized landmark features
        Returns:
            (B, T, hidden_dim) float tensor
        """
        x = self.proj(x)
        x = self.norm(x)
        return self.dropout(x)


# ============================================================================
# 2. PositionalEncoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Fixed (non-trainable) sinusoidal positional encoding, added to the
    embedded sequence so the transformer has a notion of frame order.

    Precomputed once up to max_len and registered as a buffer (not a
    Parameter) — it's not learned, it's not optimizer state, and it should
    move with the module to GPU/CPU via .to(device) like any buffer does.

    Input  : (B, T, hidden_dim), T <= max_len
    Output : (B, T, hidden_dim)  — input + positional signal
    """

    def __init__(self, hidden_dim: int, max_len: int = 150):
        super().__init__()
        pe = self._build(max_len, hidden_dim)
        # register_buffer: saved/loaded with state_dict, moved with .to(device),
        # but never touched by the optimizer.
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build(max_len: int, dim: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )                                                               # (dim/2,)
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, dim) — broadcasts over batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_dim), T <= max_len (raises if T exceeds max_len —
               fail loudly rather than silently truncating positional info)
        Returns:
            (B, T, hidden_dim)
        """
        T = x.size(1)
        if T > self.pe.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds PositionalEncoding max_len "
                f"{self.pe.size(1)}. Increase max_len at construction time."
            )
        return x + self.pe[:, :T]


# ============================================================================
# 3. TransformerEncoder (the backbone)
# ============================================================================

class TransformerEncoder(nn.Module):
    """
    Thin wrapper around nn.TransformerEncoder. This IS the CNN/GCN-free
    backbone referenced throughout the plan docs — no convolution, no
    graph operations, just self-attention + feedforward layers stacked
    n_layers deep.

    Input  : (B, T, hidden_dim), mask (B, T) bool, True = real frame
    Output : (B, T, hidden_dim) — contextualized per-frame representations

    Padding convention: nn.TransformerEncoderLayer's src_key_padding_mask
    expects True = PAD (ignore), which is the inverse of our convention
    (True = real frame). We invert it once here so every caller of this
    class uses the same "True = real frame" convention everywhere else
    in the codebase — don't leak the inverted convention outward.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        dim_feedforward = dim_feedforward or hidden_dim * 4

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, T, D) everywhere — never (T, B, D)
            norm_first=True,    # pre-LN: more stable training for deeper stacks
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, hidden_dim)
            mask: (B, T) bool or {0,1} tensor, True/1 = real frame, False/0 = padding
        Returns:
            (B, T, hidden_dim)
        """
        key_padding_mask = ~mask.bool()  # invert: True = PAD, for PyTorch's API
        return self.layers(x, src_key_padding_mask=key_padding_mask)


# ============================================================================
# 4. MaskedMeanPooling
# ============================================================================

class MaskedMeanPooling(nn.Module):
    """
    Collapses the time dimension via mean pooling, respecting the padding
    mask so padded frames (which are zeros, but could be anything) never
    contribute to the pooled representation.

    Input  : x (B, T, hidden_dim), mask (B, T) bool/float, True/1 = real frame
    Output : (B, hidden_dim)

    Guards against the all-zero-mask edge case (a clip with zero real
    frames) by clamping the denominator — this should never happen if
    upstream filtering (build_dataset.py's quality filters) is working,
    but a silent div-by-zero producing NaN is a much worse failure mode
    than an explicit clamp.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)               # (B, T, 1)
        summed = (x * mask).sum(dim=1)                   # (B, hidden_dim)
        counts = mask.sum(dim=1).clamp(min=1e-6)         # (B, 1) — avoid div-by-zero
        return summed / counts


# ============================================================================
# 5. ClassifierHead
# ============================================================================

class ClassifierHead(nn.Module):
    """
    Final projection from pooled hidden representation to class logits.

    This is specifically the WLASL ISLR (isolated sign classification)
    head. It is NOT reused for the ASLLRP continuous-signing / boundary
    detection task in Month 2 — that task needs a per-frame (not pooled)
    output for CTC + boundary loss, which will live in a separate head
    class when that work starts. Keeping this head separate (rather than
    baking classification into SignEncoder.forward directly) makes that
    swap a one-line change later, not a rewrite.

    Input  : (B, hidden_dim)
    Output : (B, n_classes) — raw logits, no softmax (use with nn.CrossEntropyLoss)
    """

    def __init__(self, hidden_dim: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


# ============================================================================
# 6. SignEncoder — wires everything together
# ============================================================================

class SignEncoder(nn.Module):
    """
    Full model: LandmarkEmbedding -> PositionalEncoding -> TransformerEncoder
    -> MaskedMeanPooling -> ClassifierHead.

    This is the Month 1 / Week 2 deliverable referenced in the plan doc.
    Defaults match the plan's stated hyperparameters (hidden_dim=384,
    n_layers=6, n_heads=8) — deliberately small relative to BERT-base,
    since landmark sequences carry far less information per timestep than
    text tokens (see plan doc's "why these hyperparameters").

    Args:
        input_dim:    per-frame feature dim. Must match configs.landmark_config.feature_dim().
                      Do not hardcode 283 here — pass it in explicitly so a config
                      change (e.g. disabling face landmarks) can't silently desync
                      the model from the data.
        hidden_dim:   transformer model dimension
        n_layers:     number of transformer encoder layers
        n_heads:      number of attention heads (hidden_dim must be divisible by this)
        n_classes:    number of WLASL gloss classes for the ISLR head
        max_len:      max sequence length for positional encoding (must be >= any T seen)
        dropout:      applied in embedding, transformer layers, and classifier head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        n_classes: int = 2000,
        max_len: int = 150,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.embedding = LandmarkEmbedding(input_dim, hidden_dim, dropout=dropout)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_len)
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads, dropout=dropout
        )
        self.pool = MaskedMeanPooling()
        self.classifier = ClassifierHead(hidden_dim, n_classes, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, input_dim)  raw landmark features
            mask: (B, T) bool/float, True/1 = real frame, False/0 = padding
        Returns:
            logits: (B, n_classes)
        """
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Input last dim {x.size(-1)} != configured input_dim {self.input_dim}. "
                f"Check that x matches configs.landmark_config.feature_dim()."
            )
        h = self.embedding(x)              # (B, T, hidden_dim)
        h = self.pos_encoding(h)           # (B, T, hidden_dim)
        h = self.encoder(h, mask)          # (B, T, hidden_dim)
        pooled = self.pool(h, mask)        # (B, hidden_dim)
        return self.classifier(pooled)     # (B, n_classes)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns pre-classifier pooled features (B, hidden_dim) instead of logits.

        Exposed separately so D can pull contextualized features for the C3
        gating head, or for ASLLRP fine-tuning later, without needing the
        WLASL-specific classifier head at all.
        """
        h = self.embedding(x)
        h = self.pos_encoding(h)
        h = self.encoder(h, mask)
        return self.pool(h, mask)


# ============================================================================
# Smoke test — runs with dummy tensors, no dataset required
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # Match landmark_config.py's contract: ~283 input dims, max_seq_len=150
    INPUT_DIM = 283
    MAX_LEN = 150
    BATCH = 4
    N_CLASSES = 2000

    print("=" * 60)
    print("SignEncoder smoke test (dummy data, no dataset required)")
    print("=" * 60)

    model = SignEncoder(
        input_dim=INPUT_DIM,
        hidden_dim=384,
        n_layers=6,
        n_heads=8,
        n_classes=N_CLASSES,
        max_len=MAX_LEN,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built. Total parameters: {n_params:,}")

    # --- Case 1: variable real-frame counts within a padded batch ---
    seq_len = 100  # < MAX_LEN, simulating a padded batch
    x = torch.randn(BATCH, seq_len, INPUT_DIM)

    real_frames = torch.tensor([100, 80, 45, 12])  # last one is very short
    mask = torch.zeros(BATCH, seq_len, dtype=torch.bool)
    for i, n in enumerate(real_frames):
        mask[i, :n] = True

    logits = model(x, mask)
    print(f"\n[Case 1] Variable-length batch")
    print(f"  input  shape: {x.shape}")
    print(f"  mask   shape: {mask.shape}")
    print(f"  logits shape: {logits.shape}  (expected: ({BATCH}, {N_CLASSES}))")
    assert logits.shape == (BATCH, N_CLASSES)

    # --- Case 2: encode() path (no classifier) ---
    feats = model.encode(x, mask)
    print(f"\n[Case 2] encode() path")
    print(f"  features shape: {feats.shape}  (expected: ({BATCH}, 384))")
    assert feats.shape == (BATCH, 384)

    # --- Case 3: full-length sequence (T == MAX_LEN) ---
    x_full = torch.randn(BATCH, MAX_LEN, INPUT_DIM)
    mask_full = torch.ones(BATCH, MAX_LEN, dtype=torch.bool)
    logits_full = model(x_full, mask_full)
    print(f"\n[Case 3] Full-length sequence (T == max_len)")
    print(f"  logits shape: {logits_full.shape}")
    assert logits_full.shape == (BATCH, N_CLASSES)

    # --- Case 4: mismatched input_dim raises clearly ---
    print(f"\n[Case 4] Wrong input_dim should raise ValueError")
    try:
        bad_x = torch.randn(BATCH, seq_len, INPUT_DIM + 1)
        model(bad_x, mask)
        raise AssertionError("Expected ValueError but none was raised")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    # --- Case 5: sequence longer than max_len raises clearly ---
    print(f"\n[Case 5] T > max_len should raise ValueError")
    try:
        too_long_x = torch.randn(BATCH, MAX_LEN + 1, INPUT_DIM)
        too_long_mask = torch.ones(BATCH, MAX_LEN + 1, dtype=torch.bool)
        model(too_long_x, too_long_mask)
        raise AssertionError("Expected ValueError but none was raised")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    # --- Case 6: gradient flows end-to-end ---
    print(f"\n[Case 6] Backward pass / gradient flow check")
    model.zero_grad()
    logits = model(x, mask)
    labels = torch.randint(0, N_CLASSES, (BATCH,))
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    n_with_grad = len(grad_norms)
    n_total = sum(1 for _ in model.parameters())
    print(f"  loss: {loss.item():.4f}")
    print(f"  params with gradients: {n_with_grad}/{n_total}")
    assert n_with_grad == n_total, "Some parameters did not receive gradients!"
    assert all(g == g for g in grad_norms), "NaN gradient detected!"  # NaN != NaN

    # --- Case 7: padding doesn't leak into pooled output ---
    print(f"\n[Case 7] Padding-invariance check")
    # Same first 12 real frames, different garbage in the padded region —
    # pooled output for that row should be identical either way.
    x_a = torch.randn(1, 50, INPUT_DIM)
    x_b = x_a.clone()
    x_b[:, 12:] = torch.randn(1, 38, INPUT_DIM) * 100  # corrupt padding region
    mask_short = torch.zeros(1, 50, dtype=torch.bool)
    mask_short[:, :12] = True
    model.eval()
    with torch.no_grad():
        feat_a = model.encode(x_a, mask_short)
        feat_b = model.encode(x_b, mask_short)
    max_diff = (feat_a - feat_b).abs().max().item()
    print(f"  max diff between identical-real-frames, different-padding: {max_diff:.2e}")
    # Note: TransformerEncoder's attention is masked so padded keys are never
    # attended to, but padded values can still affect outputs if softmax over
    # an all-masked row produces uniform attention to padding — small numerical
    # diff is expected, should NOT be large.
    assert max_diff < 1e-3, "Padding is leaking into the pooled representation!"

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)