"""
metrics.py
----------
Evaluation metrics for the WLASL ISLR task: Top-1 accuracy, Top-5 accuracy,
and confusion matrix.

Design choice: a stateful MetricTracker class, not just bare functions.
Reasoning: validate() in engine/trainer.py runs over many batches before
it can report a final number — accumulating correct/total counts per-batch
and computing the ratio once at the end is both more memory-efficient and
more numerically correct than averaging per-batch accuracies (which silently
gives wrong answers when the last batch is a different size, e.g. drop_last=False).

Confusion matrix is built as a plain (n_classes, n_classes) numpy array via
accumulation, not via sklearn, to avoid adding a new dependency for one
function — rows = true label, columns = predicted label (standard convention).
"""

from __future__ import annotations

import numpy as np
import torch


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Computes Top-k accuracy for a single batch.

    Args:
        logits: (B, n_classes) raw logits (no softmax needed — argsort is
                 invariant to monotonic transforms like softmax)
        labels: (B,) long tensor of true class ids
        k:      how many top predictions count as a "hit"

    Returns:
        float in [0, 1] — fraction of the batch where the true label is
        among the top-k predicted classes.

    Raises:
        ValueError if k > n_classes (a silently-wrong "100% Top-50 accuracy
        on a 10-class problem" is worse than a loud error here).
    """
    n_classes = logits.size(-1)
    if k > n_classes:
        raise ValueError(f"k={k} exceeds n_classes={n_classes}")
    if logits.size(0) != labels.size(0):
        raise ValueError(
            f"Batch size mismatch: logits has {logits.size(0)} rows, "
            f"labels has {labels.size(0)}."
        )

    topk_preds = logits.topk(k, dim=-1).indices            # (B, k)
    correct = topk_preds.eq(labels.unsqueeze(1)).any(dim=1)  # (B,)
    return correct.float().mean().item()


class MetricTracker:
    """
    Accumulates Top-1, Top-5 accuracy, and a confusion matrix across many
    batches (e.g. a full validation epoch), then reports final numbers.

    Usage:
        tracker = MetricTracker(n_classes=2000)
        for batch in val_loader:
            logits = model(...)
            tracker.update(logits, labels)
        results = tracker.compute()
        # results = {"top1": 0.42, "top5": 0.71, "n_samples": 1234}
        cm = tracker.confusion_matrix()   # (n_classes, n_classes) np.ndarray

    Why accumulate raw counts (not running averages of per-batch accuracy):
        Averaging per-batch accuracy is only correct if every batch is the
        same size. With drop_last=False (the default in data/dataloader.py),
        the last batch of an epoch is often smaller — naively averaging
        per-batch percentages over-weights that final small batch. Counting
        raw correct/total across the whole epoch and dividing once at the
        end is the only way to get the true epoch-level accuracy.
    """

    def __init__(self, n_classes: int, top_k: tuple[int, ...] = (1, 5)):
        self.n_classes = n_classes
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        self._correct = {k: 0 for k in self.top_k}
        self._total = 0
        self._confusion = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Accumulate stats from one batch. Call once per batch during validation.

        Args:
            logits: (B, n_classes)
            labels: (B,)
        """
        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"Batch size mismatch: logits has {logits.size(0)} rows, "
                f"labels has {labels.size(0)}."
            )
        B = logits.size(0)
        self._total += B

        for k in self.top_k:
            topk_preds = logits.topk(min(k, self.n_classes), dim=-1).indices
            hits = topk_preds.eq(labels.unsqueeze(1)).any(dim=1)
            self._correct[k] += int(hits.sum().item())

        # Confusion matrix only makes sense for top-1 predictions
        top1_preds = logits.argmax(dim=-1)
        for true_label, pred_label in zip(labels.tolist(), top1_preds.tolist()):
            self._confusion[true_label, pred_label] += 1

    def compute(self) -> dict:
        """
        Returns a dict like {"top1": 0.42, "top5": 0.71, "n_samples": 1234}.

        Returns 0.0 for accuracy keys (not NaN, not a crash) if update() was
        never called — an empty validation epoch is a bug to investigate,
        not a reason to propagate NaN into a checkpoint-selection comparison.
        """
        if self._total == 0:
            result = {f"top{k}": 0.0 for k in self.top_k}
            result["n_samples"] = 0
            return result
        result = {f"top{k}": self._correct[k] / self._total for k in self.top_k}
        result["n_samples"] = self._total
        return result

    def confusion_matrix(self) -> np.ndarray:
        """Returns the accumulated (n_classes, n_classes) confusion matrix.
        Rows = true label, columns = predicted label."""
        return self._confusion.copy()

    def per_class_accuracy(self) -> np.ndarray:
        """
        Returns (n_classes,) array of per-class top-1 recall
        (diagonal / row sum), with classes that never appeared in this
        epoch's labels reported as NaN rather than 0 — 0 would be
        indistinguishable from "appeared but always misclassified."
        """
        row_sums = self._confusion.sum(axis=1)
        diag = np.diag(self._confusion)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class = diag / row_sums
        per_class[row_sums == 0] = np.nan
        return per_class


# ============================================================================
# Smoke test — dummy tensors only
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print("metrics.py smoke test")
    print("=" * 60)

    N_CLASSES = 10

    # --- top_k_accuracy: construct a known-answer case ---
    print("\n[top_k_accuracy] Known-answer construction")
    logits = torch.zeros(4, N_CLASSES)
    labels = torch.tensor([0, 1, 2, 3])
    # Make label the argmax for samples 0,1 (top-1 hit), and rank-3 for sample 2,
    # rank-6 for sample 3 (top-5 miss)
    logits[0, 0] = 10.0                                  # top-1 hit
    logits[1, 1] = 10.0                                  # top-1 hit
    logits[2, [9, 8, 2]] = torch.tensor([10., 9., 8.])    # label rank 3 -> top5 hit, top1 miss
    logits[3, [9, 8, 7, 6, 5, 3]] = torch.tensor([10., 9., 8., 7., 6., 1.])  # label rank 6 -> top5 miss

    top1 = top_k_accuracy(logits, labels, k=1)
    top5 = top_k_accuracy(logits, labels, k=5)
    print(f"  top1 = {top1} (expected 0.5: samples 0,1 hit; 2,3 miss)")
    print(f"  top5 = {top5} (expected 0.75: samples 0,1,2 hit; 3 miss)")
    assert top1 == 0.5
    assert top5 == 0.75

    # --- k > n_classes raises ---
    print("\n[Edge case] k > n_classes should raise ValueError")
    try:
        top_k_accuracy(logits, labels, k=20)
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    # --- MetricTracker: accumulate across uneven batches ---
    print("\n[MetricTracker] Accumulation across uneven batch sizes")
    tracker = MetricTracker(n_classes=N_CLASSES)

    # Batch 1: 4 samples, all top-1 correct
    logits_b1 = torch.zeros(4, N_CLASSES)
    labels_b1 = torch.tensor([0, 1, 2, 3])
    for i, lbl in enumerate(labels_b1):
        logits_b1[i, lbl] = 10.0
    tracker.update(logits_b1, labels_b1)

    # Batch 2: 1 sample (simulates drop_last=False final small batch), top-1 WRONG
    logits_b2 = torch.zeros(1, N_CLASSES)
    labels_b2 = torch.tensor([5])
    logits_b2[0, 7] = 10.0  # predicts class 7, true is 5 -> miss
    tracker.update(logits_b2, labels_b2)

    results = tracker.compute()
    print(f"  results = {results}")
    # True accuracy: 4 correct out of 5 total = 0.8
    # A naive per-batch average would give (1.0 + 0.0) / 2 = 0.5 -- WRONG
    assert abs(results["top1"] - 0.8) < 1e-6, (
        f"Expected weighted top1=0.8, got {results['top1']} "
        f"(this would indicate per-batch averaging instead of count accumulation)"
    )
    assert results["n_samples"] == 5
    print("  Correct: weighted accumulation (0.8), not naive batch-average (0.5).")

    # --- Confusion matrix ---
    print("\n[Confusion matrix]")
    cm = tracker.confusion_matrix()
    print(f"  shape: {cm.shape} (expected ({N_CLASSES}, {N_CLASSES}))")
    print(f"  total counted: {cm.sum()} (expected 5)")
    assert cm.shape == (N_CLASSES, N_CLASSES)
    assert cm.sum() == 5
    assert cm[5, 7] == 1, "Expected the true=5,pred=7 miss to be recorded"
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1 and cm[3, 3] == 1

    # --- per_class_accuracy with unseen classes ---
    print("\n[per_class_accuracy] Classes never seen should be NaN, not 0")
    pca = tracker.per_class_accuracy()
    print(f"  class 0 acc: {pca[0]} (expected 1.0)")
    print(f"  class 9 acc: {pca[9]} (expected NaN, never appeared in labels)")
    assert pca[0] == 1.0
    assert np.isnan(pca[9])

    # --- empty tracker doesn't crash / doesn't return NaN ---
    print("\n[Edge case] Empty tracker (no updates) should return 0.0, not NaN/crash")
    empty_tracker = MetricTracker(n_classes=N_CLASSES)
    empty_results = empty_tracker.compute()
    print(f"  empty results: {empty_results}")
    assert empty_results["top1"] == 0.0
    assert empty_results["n_samples"] == 0

    print("\n" + "=" * 60)
    print("All metrics.py smoke tests passed.")
    print("=" * 60)