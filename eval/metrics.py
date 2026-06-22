"""Evaluation metrics for ARC-AGI-3 agent performance.

Tracks per-game/level scores, action efficiency vs. human baseline,
and world-model prediction quality (AUC, calibration).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class EvalMetrics:
    """Tracks and computes evaluation metrics.

    Records per-step data during gameplay and computes aggregate
    metrics: score progression, action distribution, efficiency,
    state coverage, and model quality.

    Attributes:
        steps: List of per-step records.
    """

    def __init__(self) -> None:
        """Initialize an empty metrics tracker."""
        self.steps: list[dict] = []

    def record_step(
        self,
        score: float,
        action: str,
        **extra: object,
    ) -> None:
        """Record a single step's metrics.

        Args:
            score: Current game score.
            action: Action taken.
            **extra: Additional metrics (novelty, confidence, etc.).
        """
        self.steps.append({
            "score": score,
            "action": action,
            **extra,
        })

    def compute(self) -> dict:
        """Compute aggregate metrics.

        Returns:
            Dict with:
            - total_steps: Number of actions taken.
            - final_score: Last recorded score.
            - action_distribution: Dict of action → count.
            - unique_actions: Number of distinct actions used.
            - score_delta: Final score - initial score.
        """
        if not self.steps:
            return {
                "total_steps": 0,
                "final_score": 0.0,
                "action_distribution": {},
                "unique_actions": 0,
                "score_delta": 0.0,
            }

        action_counts: dict[str, int] = defaultdict(int)
        for step in self.steps:
            action_counts[step["action"]] += 1

        scores = [s["score"] for s in self.steps]
        initial_score = scores[0] if scores else 0.0
        final_score = scores[-1] if scores else 0.0

        return {
            "total_steps": len(self.steps),
            "final_score": final_score,
            "action_distribution": dict(action_counts),
            "unique_actions": len(action_counts),
            "score_delta": final_score - initial_score,
        }


def compute_efficiency(metrics: dict, max_actions: int) -> dict:
    """Compute action efficiency relative to the action budget.

    ARC-AGI-3 scores are squared and action budgets are capped at
    ~5× human median. This function computes how efficiently the agent
    used its budget.

    Args:
        metrics: Output of EvalMetrics.compute().
        max_actions: The action budget for this level.

    Returns:
        Dict with:
        - budget_used: Fraction of budget used (0–1).
        - actions_per_score: Actions per unit of score gained.
        - efficiency_score: Composite efficiency metric (0–1, higher is better).
    """
    total_steps = metrics.get("total_steps", 0)
    score_delta = metrics.get("score_delta", 0.0)

    budget_used = total_steps / max(1, max_actions)
    actions_per_score = total_steps / max(abs(score_delta), 0.001)

    # Efficiency: high score with low budget usage is best
    if score_delta > 0:
        efficiency_score = score_delta * (1.0 - 0.5 * budget_used)
    else:
        efficiency_score = 0.0

    return {
        "budget_used": budget_used,
        "actions_per_score": actions_per_score,
        "efficiency_score": min(efficiency_score, 1.0),
    }


def compute_change_prediction_auc(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute AUC for the world model's change-prediction head.

    Args:
        predictions: Predicted probabilities (float array).
        labels: Binary labels (0 or 1).

    Returns:
        ROC AUC score (0–1).
    """
    if len(predictions) == 0 or len(np.unique(labels)) < 2:
        return 0.5

    # Sort by prediction descending
    order = np.argsort(-predictions)
    labels_sorted = labels[order]

    # Compute ROC AUC via rank-based formula
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank sum
    ranks = np.zeros(len(predictions))
    for i, idx in enumerate(order):
        ranks[idx] = len(predictions) - i

    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return float(auc)


def compare_agents(
    baseline: dict,
    candidate: dict,
) -> dict:
    """Compare two agents' evaluation results.

    Args:
        baseline: Baseline agent results.
        candidate: Candidate agent results.

    Returns:
        Dict with per-game and aggregate comparisons.
    """
    comparison = {
        "games": {},
        "aggregate": {},
    }

    for game_id in baseline.get("games", {}):
        if game_id in candidate.get("games", {}):
            b = baseline["games"][game_id]
            c = candidate["games"][game_id]
            comparison["games"][game_id] = {
                "score_delta": c["score"] - b["score"],
                "action_delta": c["total_actions"] - b["total_actions"],
                "win_delta": c["levels_won"] - b["levels_won"],
            }

    b_agg = baseline.get("aggregate", {})
    c_agg = candidate.get("aggregate", {})
    comparison["aggregate"] = {
        "score_delta": c_agg.get("total_score", 0) - b_agg.get("total_score", 0),
        "win_rate_delta": c_agg.get("win_rate", 0) - b_agg.get("win_rate", 0),
        "action_delta": c_agg.get("total_actions", 0) - b_agg.get("total_actions", 0),
    }

    return comparison
