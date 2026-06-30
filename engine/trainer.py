"""
engine/trainer.py
------------------
The training engine. Owns the actual loop mechanics so train.py (or a
notebook, or a future CLI) stays a thin "wire everything together and call
trainer.fit()" script, not a 200-line god-function.

Contains:
    train_epoch()      — one pass over the train DataLoader
    validate()          — one pass over the val DataLoader (no grad, eval mode)
    save_checkpoint()   — saves model/optimizer/scheduler state + metadata
    load_checkpoint()   — companion loader (you'll want this in Month 2 when
                          D attaches the C3 gating head to your checkpoint)
    EarlyStopping        — stateful early-stopping tracker
    Trainer              — wires the above into a fit() loop with TensorBoard
                          logging, matching the plan's Week 2-3 training spec
                          (AdamW, cosine annealing, grad clipping, 50 epochs)

Design choice: train_epoch() and validate() are plain functions, not methods
buried in Trainer, specifically so they're independently callable/testable
(e.g. from a notebook or a unit test) without instantiating the full Trainer.
Trainer.fit() just calls them in a loop and adds checkpointing/logging/
early-stopping on top.

This file is fully testable with a dummy model + dummy DataLoaders — no
WLASL or GPU required. See `if __name__ == "__main__"` at the bottom.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import MetricTracker

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional for the smoke test environment
    SummaryWriter = None


# ============================================================================
# train_epoch / validate — the two core loop functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    log_fn=print,
) -> dict:
    """
    Runs one full pass over the training DataLoader.

    Args:
        model:          the SignEncoder (or any model with forward(x, mask) -> logits)
        loader:         train DataLoader yielding (landmarks, mask, labels)
        optimizer:      e.g. AdamW
        criterion:      e.g. ClassificationLoss from models/losses.py
        device:         torch.device to move batches to
        grad_clip_norm: max_norm for torch.nn.utils.clip_grad_norm_ (per the
                        plan's "gradient clipping prevents a single bad batch
                        from blowing up your weights")
        log_fn:         callable for progress messages (print by default;
                        pass a logger.info-style function in real use)

    Returns:
        dict: {"loss": avg_train_loss, "n_batches": int, "time_s": float}

    Note on empty loader: returns loss=0.0 rather than raising ZeroDivisionError
    if the loader has zero batches — this should never happen with real data,
    but a clean return is better than a crash mid-epoch-loop for what's
    ultimately a data-pipeline bug, not a trainer bug.
    """
    model.train()
    t_start = time.perf_counter()

    total_loss = 0.0
    n_batches = 0

    for batch_idx, (landmarks, mask, labels) in enumerate(loader):
        landmarks = landmarks.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(landmarks, mask)
        loss = criterion(logits, labels)

        if torch.isnan(loss):
            raise FloatingPointError(
                f"NaN loss at batch {batch_idx}. Common causes: bad "
                f"normalization upstream, exploding gradients before "
                f"clipping kicks in, or a label/data misalignment. "
                f"Stop and investigate — do not silently skip this batch."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    elapsed = time.perf_counter() - t_start
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return {"loss": avg_loss, "n_batches": n_batches, "time_s": elapsed}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    top_k: tuple[int, ...] = (1, 5),
) -> dict:
    """
    Runs one full pass over a val/test DataLoader in eval mode, no gradients.

    Args:
        model, loader, criterion, device: same as train_epoch
        n_classes: needed to size the MetricTracker's confusion matrix
        top_k:     which top-k accuracies to compute (default matches the
                   plan's Top-1/Top-5 requirement)

    Returns:
        dict: {"loss": avg_val_loss, "top1": float, "top5": float,
               "n_samples": int, "time_s": float}

    Decorated with @torch.no_grad() at the function level (not just wrapping
    the loop body) so it's impossible to accidentally leave a gradient-
    tracked op in here later and silently bloat memory during validation.
    """
    model.eval()
    t_start = time.perf_counter()

    tracker = MetricTracker(n_classes=n_classes, top_k=top_k)
    total_loss = 0.0
    n_batches = 0

    for landmarks, mask, labels in loader:
        landmarks = landmarks.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(landmarks, mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1
        tracker.update(logits, labels)

    elapsed = time.perf_counter() - t_start
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    metrics = tracker.compute()
    metrics["loss"] = avg_loss
    metrics["time_s"] = elapsed
    metrics["_tracker"] = tracker  # callers can pull confusion_matrix() if needed
    return metrics


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    epoch: int = 0,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> None:
    """
    Saves a checkpoint with everything needed to resume training OR for
    another team member (D, per the plan) to load just the model weights.

    Args:
        path:      output .pt path; parent dirs created if missing
        model:     saved as model.state_dict()
        optimizer: optional — omit when saving a "frozen, share with team" checkpoint
        scheduler: optional
        epoch:     current epoch number, for logging/resume
        metrics:   optional dict of metrics at save time (e.g. val top1/top5) —
                   stored alongside the weights so "which checkpoint had what
                   accuracy" never depends on a separate log file surviving
        extra:     optional dict for anything else (e.g. config_hash from
                   configs/landmark_config.py, so a loader can verify the
                   checkpoint matches the tensor contract it expects)

    Note: `metrics` may contain a MetricTracker object under "_tracker" (see
    validate()) which is NOT picklable in a portable way — this function
    strips that key automatically so checkpoints stay loadable across
    environments.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_metrics = None
    if metrics is not None:
        clean_metrics = {k: v for k, v in metrics.items() if k != "_tracker"}

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": clean_metrics,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """
    Loads a checkpoint saved by save_checkpoint() back into model (and
    optionally optimizer/scheduler) in place.

    Returns the full payload dict (epoch, metrics, extra) so callers can
    inspect what they loaded — e.g. D checking `extra["config_hash"]`
    matches their expected tensor contract before trusting the weights.

    Raises FileNotFoundError with a clear message rather than torch's
    default (less obvious) error if the path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")

    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


# ============================================================================
# Early stopping
# ============================================================================

class EarlyStopping:
    """
    Tracks validation metric across epochs and signals when to stop.

    Usage:
        stopper = EarlyStopping(patience=5, mode="max")  # e.g. for val top1
        for epoch in range(50):
            ...
            val_metrics = validate(...)
            if stopper.step(val_metrics["top1"]):
                print("Early stopping triggered")
                break
        if stopper.is_best:   # check after each .step() call
            save_checkpoint(...)   # this epoch is the new best

    Args:
        patience: number of epochs with no improvement before stopping
        mode:     "max" for metrics where higher is better (accuracy),
                  "min" for metrics where lower is better (loss)
        min_delta: minimum change to count as an improvement (guards
                   against stopping decisions being noise-sensitive on
                   tiny floating point fluctuations)
    """

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_value: float | None = None
        self.epochs_without_improvement = 0
        self.is_best = False   # set by the most recent .step() call

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        else:
            return value < self.best_value - self.min_delta

    def step(self, value: float) -> bool:
        """
        Call once per epoch with the metric to track.

        Returns:
            True if training should stop now (patience exhausted).
        Also sets self.is_best to True/False for this call, so callers can
        decide whether to save a checkpoint right after calling .step().
        """
        if self._improved(value):
            self.best_value = value
            self.epochs_without_improvement = 0
            self.is_best = True
        else:
            self.epochs_without_improvement += 1
            self.is_best = False

        return self.epochs_without_improvement >= self.patience


# ============================================================================
# Trainer — wires everything into a fit() loop
# ============================================================================

class Trainer:
    """
    High-level orchestrator. Wires train_epoch/validate/save_checkpoint/
    EarlyStopping into the fit() loop matching the plan's training spec:
    AdamW + cosine annealing + grad clipping + TensorBoard logging.

    train.py should be little more than:
        model = SignEncoder(...)
        loaders = build_dataloaders(...)
        trainer = Trainer(model, loaders, n_classes=2000, ...)
        trainer.fit(n_epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_classes: int,
        device: torch.device | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
        criterion: nn.Module | None = None,
        log_dir: str | Path = "runs/sign_encoder",
        checkpoint_dir: str | Path = "checkpoints",
        early_stopping_patience: int = 8,
    ):
        from models.losses import ClassificationLoss  # local import avoids hard
                                                        # dependency for callers
                                                        # that bring their own loss

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_classes = n_classes
        self.grad_clip_norm = grad_clip_norm

        self.criterion = criterion or ClassificationLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # T_max set when fit() is called, since it depends on n_epochs
        self.scheduler = None

        self.checkpoint_dir = Path(checkpoint_dir)
        self.early_stopper = EarlyStopping(patience=early_stopping_patience, mode="max")

        self.writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter else None
        if self.writer is None:
            print("[Trainer] WARNING: tensorboard not installed — skipping TB logging.")

        self.history: list[dict] = []  # one dict per epoch, for callers/tests to inspect

    def fit(self, n_epochs: int = 50) -> list[dict]:
        """
        Runs the full training loop for n_epochs (or until early stopping
        triggers). Returns self.history (also stored on the instance).
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs
        )

        for epoch in range(1, n_epochs + 1):
            train_stats = train_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion,
                self.device, grad_clip_norm=self.grad_clip_norm,
            )
            val_stats = validate(
                self.model, self.val_loader, self.criterion, self.device,
                n_classes=self.n_classes,
            )
            self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"[epoch {epoch}/{n_epochs}] "
                f"train_loss={train_stats['loss']:.4f}  "
                f"val_loss={val_stats['loss']:.4f}  "
                f"val_top1={val_stats['top1']:.4f}  "
                f"val_top5={val_stats['top5']:.4f}  "
                f"lr={lr_now:.2e}"
            )

            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_stats["loss"], epoch)
                self.writer.add_scalar("loss/val", val_stats["loss"], epoch)
                self.writer.add_scalar("accuracy/val_top1", val_stats["top1"], epoch)
                self.writer.add_scalar("accuracy/val_top5", val_stats["top5"], epoch)
                self.writer.add_scalar("lr", lr_now, epoch)

            should_stop = self.early_stopper.step(val_stats["top1"])

            if self.early_stopper.is_best:
                save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    self.model, self.optimizer, self.scheduler,
                    epoch=epoch,
                    metrics={"top1": val_stats["top1"], "top5": val_stats["top5"],
                             "val_loss": val_stats["loss"]},
                )

            self.history.append({
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "val_top1": val_stats["top1"],
                "val_top5": val_stats["top5"],
                "lr": lr_now,
            })

            if should_stop:
                print(
                    f"[Trainer] Early stopping at epoch {epoch} "
                    f"(no val_top1 improvement for {self.early_stopper.patience} epochs). "
                    f"Best val_top1={self.early_stopper.best_value:.4f}."
                )
                break

        # Always save a final checkpoint too, separate from "best", so you
        # have both "best so far" and "where training actually ended."
        save_checkpoint(
            self.checkpoint_dir / "last.pt",
            self.model, self.optimizer, self.scheduler,
            epoch=self.history[-1]["epoch"],
            metrics={"top1": self.history[-1]["val_top1"], "top5": self.history[-1]["val_top5"]},
        )

        if self.writer is not None:
            self.writer.close()

        return self.history


# ============================================================================
# Smoke test — dummy model + dummy data, no WLASL/GPU required
# ============================================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.transformer import SignEncoder
    from models.losses import ClassificationLoss

    print("=" * 60)
    print("trainer.py smoke test (dummy model + synthetic data)")
    print("=" * 60)

    torch.manual_seed(0)
    device = torch.device("cpu")
    D, N_CLASSES, MAX_LEN = 64, 5, 30

    # --- Build a tiny fake "dataset" directly as tensors (skip data/dataset.py
    # here to keep this file's smoke test self-contained and fast) ---
    def make_loader(n_samples, batch_size=4):
        from torch.utils.data import TensorDataset
        x = torch.randn(n_samples, MAX_LEN, D)
        mask = torch.ones(n_samples, MAX_LEN, dtype=torch.bool)
        y = torch.randint(0, N_CLASSES, (n_samples,))
        return DataLoader(TensorDataset(x, mask, y), batch_size=batch_size, shuffle=True)

    train_loader = make_loader(40)
    val_loader = make_loader(16)

    model = SignEncoder(input_dim=D, hidden_dim=32, n_layers=2, n_heads=2,
                         n_classes=N_CLASSES, max_len=MAX_LEN)
    criterion = ClassificationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\n[1] train_epoch() runs without error and returns sane stats")
    stats = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"  {stats}")
    assert stats["n_batches"] == 10  # 40 samples / batch 4
    assert stats["loss"] > 0

    print("\n[2] validate() runs without error and returns sane stats")
    val_stats = validate(model, val_loader, criterion, device, n_classes=N_CLASSES)
    print(f"  loss={val_stats['loss']:.4f} top1={val_stats['top1']:.4f} top5={val_stats['top5']:.4f}")
    assert 0.0 <= val_stats["top1"] <= 1.0
    assert 0.0 <= val_stats["top5"] <= 1.0
    assert val_stats["top5"] >= val_stats["top1"], "top5 should never be lower than top1"

    print("\n[3] Checkpoint save/load round-trip preserves weights exactly")
    tmp_dir = Path(tempfile.mkdtemp())
    ckpt_path = tmp_dir / "test.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch=3, metrics=val_stats,
                     extra={"config_hash": "abc123"})
    assert ckpt_path.exists()

    model2 = SignEncoder(input_dim=D, hidden_dim=32, n_layers=2, n_heads=2,
                          n_classes=N_CLASSES, max_len=MAX_LEN)
    payload = load_checkpoint(ckpt_path, model2)
    print(f"  loaded epoch={payload['epoch']}, extra={payload['extra']}")
    assert payload["epoch"] == 3
    assert payload["extra"]["config_hash"] == "abc123"
    assert "_tracker" not in payload["metrics"], "MetricTracker should be stripped before saving"

    # Confirm weights actually match after load
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Loaded weights don't match saved weights!"
    print("  Weights match exactly after save/load round-trip.")

    print("\n[Edge case] load_checkpoint on missing path raises FileNotFoundError")
    try:
        load_checkpoint(tmp_dir / "nonexistent.pt", model2)
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  Correctly raised: {e}")

    print("\n[4] EarlyStopping: stops after patience epochs with no improvement")
    stopper = EarlyStopping(patience=3, mode="max")
    sequence = [0.5, 0.6, 0.6, 0.6, 0.6]  # improves once, then plateaus for 4 epochs
    stopped_at = None
    for i, v in enumerate(sequence):
        should_stop = stopper.step(v)
        print(f"  epoch {i}: value={v}, is_best={stopper.is_best}, stop={should_stop}")
        if should_stop:
            stopped_at = i
            break
    assert stopped_at == 4, f"Expected stop at index 4 (3 non-improving epochs after the best), got {stopped_at}"
    assert stopper.best_value == 0.6

    print("\n[Edge case] EarlyStopping invalid mode raises ValueError")
    try:
        EarlyStopping(mode="sideways")
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    print("\n[5] NaN loss raises FloatingPointError instead of silently continuing")
    class NaNLoss(nn.Module):
        def forward(self, logits, labels, mask=None):
            return torch.tensor(float("nan"))
    try:
        train_epoch(model, train_loader, optimizer, NaNLoss(), device)
        raise AssertionError("Expected FloatingPointError")
    except FloatingPointError as e:
        print(f"  Correctly raised: {e}")

    print("\n[6] Full Trainer.fit() smoke test — 3 epochs, tiny model")
    trainer = Trainer(
        model=SignEncoder(input_dim=D, hidden_dim=32, n_layers=2, n_heads=2,
                           n_classes=N_CLASSES, max_len=MAX_LEN),
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=N_CLASSES,
        device=device,
        log_dir=tmp_dir / "runs",
        checkpoint_dir=tmp_dir / "checkpoints",
        early_stopping_patience=10,
    )
    history = trainer.fit(n_epochs=3)
    print(f"  history length: {len(history)} (expected 3)")
    assert len(history) == 3
    assert (tmp_dir / "checkpoints" / "last.pt").exists()
    print("  Trainer.fit() completed, checkpoints written.")

    shutil.rmtree(tmp_dir)

    print("\n" + "=" * 60)
    print("All trainer.py smoke tests passed.")
    print("=" * 60)