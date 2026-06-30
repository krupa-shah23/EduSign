"""
models/losses.py
-----------------
Loss functions for the project. Month 1 only needs classification loss
(WLASL ISLR task). CTC + boundary-detection loss are stubbed here as
documented placeholders for Month 2 (ASLLRP continuous signing) so the
import path is stable now and nobody has to refactor callers later.

Month 1 — IMPLEMENTED:
    ClassificationLoss   — thin wrapper around nn.CrossEntropyLoss

Month 2 — STUBS ONLY (raise NotImplementedError if called):
    CTCLoss               — for continuous-signing sequence labeling
    BoundaryLoss          — for sign-boundary detection
    JointCTCBoundaryLoss  — combines the two with a λ weight (per the plan's
                             "joint CTC + boundary-detection loss, λ-weighted,
                             sweep 0.1/0.3/0.5/0.7" instruction)

Why stub these now instead of waiting: the training engine (engine/trainer.py)
needs a stable loss interface today — train_epoch() shouldn't care whether
it's calling ClassificationLoss or JointCTCBoundaryLoss, as long as both
expose the same forward(outputs, targets, mask) -> scalar tensor signature.
Defining that interface now means Month 2 plugs in without touching the
trainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """
    Month 1 loss: standard cross-entropy over WLASL gloss classes.

    Thin wrapper (not just bare nn.CrossEntropyLoss) so the call signature
    matches the shared interface every loss in this file uses:
        forward(logits, labels, mask=None) -> scalar tensor

    `mask` is accepted but unused here — classification operates on the
    already-pooled (B, n_classes) logits, which have no time dimension left
    to mask. It's in the signature only so train_epoch() can call every
    loss type identically without an if/else branch.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, n_classes) raw logits from ClassifierHead
            labels: (B,) long tensor of gloss class ids
            mask:   unused, accepted for interface compatibility
        Returns:
            scalar loss tensor
        """
        del mask  # explicitly unused — see class docstring
        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"Batch size mismatch: logits has {logits.size(0)} rows, "
                f"labels has {labels.size(0)}."
            )
        return self.criterion(logits, labels)


# ============================================================================
# Month 2 stubs — NOT implemented yet, documented placeholders only
# ============================================================================

class CTCLoss(nn.Module):
    """
    STUB — Month 2 (ASLLRP continuous signing).

    Will wrap nn.CTCLoss for frame-level gloss sequence labeling on
    continuous (multi-sign) clips, where output length != target length
    and alignment is learned rather than given.

    Not implemented in Month 1 because Month 1's task (WLASL ISLR) is
    single-label classification per clip, not sequence labeling — there's
    no alignment problem to solve yet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "CTCLoss is a Month 2 component (ASLLRP continuous signing). "
            "Month 1 only needs ClassificationLoss for WLASL ISLR."
        )


class BoundaryLoss(nn.Module):
    """
    STUB — Month 2 (ASLLRP continuous signing).

    Will supervise the per-frame sign-boundary prediction (start/end of
    each sign within a continuous sequence) — likely binary cross-entropy
    or focal loss over a per-frame boundary indicator, evaluated with the
    ±2 frame (~67ms) tolerance window mentioned in the team plan.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "BoundaryLoss is a Month 2 component (ASLLRP continuous signing). "
            "Month 1 only needs ClassificationLoss for WLASL ISLR."
        )


class JointCTCBoundaryLoss(nn.Module):
    """
    STUB — Month 2 (ASLLRP continuous signing).

    Will combine CTCLoss and BoundaryLoss with a λ weight:
        total = ctc_loss + lambda_weight * boundary_loss
    per the plan's instruction to sweep λ in {0.1, 0.3, 0.5, 0.7}.

    Kept here (not in a separate file) so the loss module's public surface
    is one import (`from models.losses import ...`) regardless of which
    month's loss is in use.
    """

    def __init__(self, lambda_weight: float = 0.3, *args, **kwargs):
        super().__init__()
        self.lambda_weight = lambda_weight
        raise NotImplementedError(
            "JointCTCBoundaryLoss is a Month 2 component (ASLLRP continuous "
            "signing). Month 1 only needs ClassificationLoss for WLASL ISLR. "
            "Draft this in Month 1 Week 4 per the plan (test on dummy data), "
            "implement for real in Month 2."
        )


# ============================================================================
# Smoke test — dummy tensors only
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("losses.py smoke test")
    print("=" * 60)

    B, N_CLASSES = 8, 50
    logits = torch.randn(B, N_CLASSES, requires_grad=True)
    labels = torch.randint(0, N_CLASSES, (B,))

    loss_fn = ClassificationLoss()
    loss = loss_fn(logits, labels)
    print(f"\n[ClassificationLoss] loss = {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be a scalar"
    loss.backward()
    assert logits.grad is not None and not torch.isnan(logits.grad).any()
    print("  Backward pass OK, no NaN gradients.")

    # Label smoothing variant
    loss_fn_smooth = ClassificationLoss(label_smoothing=0.1)
    loss_smooth = loss_fn_smooth(logits.detach().requires_grad_(), labels)
    print(f"\n[ClassificationLoss with label_smoothing=0.1] loss = {loss_smooth.item():.4f}")

    # Batch size mismatch should raise
    print(f"\n[Edge case] Batch size mismatch should raise ValueError")
    try:
        loss_fn(torch.randn(8, N_CLASSES), torch.randint(0, N_CLASSES, (5,)))
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    # Stubs should raise NotImplementedError, not silently do nothing
    print(f"\n[Month 2 stubs] Should raise NotImplementedError when instantiated")
    for cls_name, cls in [("CTCLoss", CTCLoss), ("BoundaryLoss", BoundaryLoss),
                           ("JointCTCBoundaryLoss", JointCTCBoundaryLoss)]:
        try:
            cls()
            raise AssertionError(f"Expected NotImplementedError from {cls_name}")
        except NotImplementedError as e:
            print(f"  {cls_name}: correctly raised NotImplementedError")

    print("\n" + "=" * 60)
    print("All losses.py smoke tests passed.")
    print("=" * 60)