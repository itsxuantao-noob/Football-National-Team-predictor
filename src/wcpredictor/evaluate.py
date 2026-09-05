"""Scoring helpers and calibration diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from .config import CLASSES


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between the predicted distribution and the one-hot truth."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def score(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Log loss (the metric that matters for probabilities), accuracy and Brier score."""
    labels = list(range(len(CLASSES)))
    return {
        "log_loss": float(log_loss(y_true, probs, labels=labels)),
        "accuracy": float(accuracy_score(y_true, probs.argmax(axis=1))),
        "brier": multiclass_brier(y_true, probs),
    }


def baseline_scores(y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, dict[str, float]]:
    """Two reference points: always predict the majority class, and predict the training prior."""
    n_classes = len(CLASSES)
    prior = np.bincount(y_train, minlength=n_classes) / len(y_train)

    prior_probs = np.tile(prior, (len(y_eval), 1))
    majority_probs = np.full((len(y_eval), n_classes), 1e-15)
    majority_probs[:, int(prior.argmax())] = 1 - 1e-15 * (n_classes - 1)

    return {
        "class_prior": score(y_eval, prior_probs),
        "majority_class": {
            "log_loss": float("nan"),
            "accuracy": float(np.mean(y_eval == prior.argmax())),
            "brier": multiclass_brier(y_eval, majority_probs),
        },
    }


def reliability_table(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_name: str = "win",
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Bin predictions for one class and compare predicted vs observed frequency."""
    idx = CLASSES.index(class_name)
    predicted = probs[:, idx]
    observed = (y_true == idx).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(predicted, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_predicted": float(predicted[mask].mean()),
                "observed_frequency": float(observed[mask].mean()),
            }
        )
    return rows


def plot_calibration(
    y_true: np.ndarray,
    probs: np.ndarray,
    path: Path,
    title: str = "Calibration",
    n_bins: int = 10,
) -> Path | None:
    """Write a reliability diagram for all three classes. Returns None if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    for class_name in CLASSES:
        rows = reliability_table(y_true, probs, class_name, n_bins)
        ax.plot(
            [r["mean_predicted"] for r in rows],
            [r["observed_frequency"] for r in rows],
            marker="o",
            label=class_name,
        )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
