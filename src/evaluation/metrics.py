"""Quantitative evaluation metrics — accuracy, F1, confusion matrix, statistical tests."""

import numpy as np

VERDICT_CLASSES = ["SUPPORTED", "UNSUPPORTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"]


def compute_verdict_accuracy(results: list[dict], expected: list[dict]) -> dict:
    """Compute verdict accuracy, per-class precision/recall, and macro-F1.

    Args:
        results: List of pipeline output dicts with 'verdict' key.
        expected: List of dicts with 'expected_verdict' key.

    Returns:
        Dict with 'accuracy', 'macro_f1', 'per_class' (precision, recall, f1),
        and 'confusion_matrix'.
    """
    predicted = [r["verdict"] for r in results]
    actual = [e["expected_verdict"] for e in expected]

    # Overall accuracy
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    accuracy = correct / len(actual) if actual else 0.0

    # Per-class precision/recall/F1
    per_class = {}
    f1_scores = []
    for cls in VERDICT_CLASSES:
        tp = sum(1 for p, a in zip(predicted, actual) if p == cls and a == cls)
        fp = sum(1 for p, a in zip(predicted, actual) if p == cls and a != cls)
        fn = sum(1 for p, a in zip(predicted, actual) if p != cls and a == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for a in actual if a == cls),
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Confusion matrix
    confusion = {a: {p: 0 for p in VERDICT_CLASSES} for a in VERDICT_CLASSES}
    for p, a in zip(predicted, actual):
        if a in confusion and p in confusion[a]:
            confusion[a][p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": len(actual),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def mcnemar_test(results_a: list[str], results_b: list[str], expected: list[str]) -> dict:
    """McNemar's test for paired comparison of two systems on the same claims.

    Args:
        results_a: Verdicts from system A.
        results_b: Verdicts from system B.
        expected: Ground-truth verdicts.

    Returns:
        Dict with 'statistic', 'p_value', 'significant' (at p < 0.05).
    """
    # Build contingency: b = A correct & B wrong, c = A wrong & B correct
    b = 0
    c = 0
    for a_pred, b_pred, true in zip(results_a, results_b, expected):
        a_ok = a_pred == true
        b_ok = b_pred == true
        if a_ok and not b_ok:
            b += 1
        elif not a_ok and b_ok:
            c += 1

    if b + c == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b_count": b,
            "c_count": c,
            "note": "No discordant pairs",
        }

    # McNemar's statistic with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution (1 df)
    from scipy.stats import chi2

    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "b_count": b,
        "c_count": c,
    }


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Compute bootstrap confidence interval for a metric.

    Args:
        scores: List of per-claim scores.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.

    Returns:
        Dict with 'mean', 'lower', 'upper', 'ci_level', 'std'.
    """
    scores_arr = np.array(scores, dtype=float)
    n = len(scores_arr)

    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(scores_arr, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))

    return {
        "mean": round(float(scores_arr.mean()), 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "ci_level": ci,
        "std": round(float(boot_means.std()), 4),
    }
