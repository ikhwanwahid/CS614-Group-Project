"""Quantitative evaluation metrics — accuracy, F1, confusion matrix, statistical tests."""


def compute_verdict_accuracy(results: list[dict], expected: list[dict]) -> dict:
    """Compute verdict accuracy, per-class precision/recall, and macro-F1.

    Args:
        results: List of pipeline output dicts with 'verdict' key.
        expected: List of dicts with 'expected_verdict' key.

    Returns:
        Dict with 'accuracy', 'macro_f1', 'per_class' (precision, recall, f1),
        and 'confusion_matrix'.
    """
    raise NotImplementedError(
        "Verdict accuracy metrics not yet implemented — Eval lead (Member 6).\n"
        "Approach: compute confusion matrix, per-class precision/recall/F1, "
        "macro-F1 using sklearn or manual computation."
    )


def mcnemar_test(results_a: list[str], results_b: list[str], expected: list[str]) -> dict:
    """McNemar's test for paired comparison of two systems on the same claims.

    Args:
        results_a: Verdicts from system A.
        results_b: Verdicts from system B.
        expected: Ground-truth verdicts.

    Returns:
        Dict with 'statistic', 'p_value', 'significant' (at p < 0.05).
    """
    raise NotImplementedError(
        "McNemar's test not yet implemented — Eval lead (Member 6)."
    )


def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence interval for a metric.

    Args:
        scores: List of per-claim scores.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.

    Returns:
        Dict with 'mean', 'lower', 'upper', 'ci_level'.
    """
    raise NotImplementedError(
        "Bootstrap CI not yet implemented — Eval lead (Member 6)."
    )
