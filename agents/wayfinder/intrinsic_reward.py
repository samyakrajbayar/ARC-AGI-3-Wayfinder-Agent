"""Intrinsic reward module — Module D.

Combines extrinsic score delta, graph novelty, and world-model
prediction error into a single scalar utility. Weights are
configurable via a YAML config file (not hard-coded).

The utility function guides both the reactive policy (via action
selection) and the planner (as the value function for rollouts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for the intrinsic reward module.

    Attributes:
        w_extrinsic: Weight for extrinsic score delta.
        w_novelty: Weight for graph novelty signal.
        w_curiosity: Weight for world-model prediction error (curiosity).
        novelty_decay: Multiplicative decay per visit (reduces novelty
            of repeatedly-visited states faster).
        curiosity_clip: Maximum curiosity value (prevents runaway).
        score_baseline: Baseline score for normalizing extrinsic signal.
    """

    w_extrinsic: float = 1.0
    w_novelty: float = 0.3
    w_curiosity: float = 0.2
    novelty_decay: float = 0.95
    curiosity_clip: float = 10.0
    score_baseline: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RewardConfig":
        """Load config from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            RewardConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("intrinsic_reward", {}))

    @classmethod
    def default(cls) -> "RewardConfig":
        """Return default configuration."""
        return cls()


class IntrinsicReward:
    """Computes intrinsic reward combining extrinsic + novelty + curiosity.

    The utility function is:
        utility = w_extrinsic * Δscore + w_novelty * novelty + w_curiosity * curiosity

    This maps to the "Goal-setting" benchmark capability — the agent
    learns to value states that increase score, explore new territory,
    and reduce its model's prediction error.

    Attributes:
        config: Reward configuration with tunable weights.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Initialize the intrinsic reward module.

        Args:
            config: Reward configuration. If None, uses defaults.
        """
        self.config = config or RewardConfig.default()
        self._running_baseline = self.config.score_baseline
        self._update_count = 0

        logger.info(
            "IntrinsicReward initialized (w_ext=%.2f, w_nov=%.2f, w_cur=%.2f)",
            self.config.w_extrinsic,
            self.config.w_novelty,
            self.config.w_curiosity,
        )

    def compute(
        self,
        extrinsic_delta: float = 0.0,
        novelty: float = 0.0,
        prediction_error: float = 0.0,
        **kwargs: float,
    ) -> float:
        """Compute the intrinsic utility for a state transition.

        Args:
            extrinsic_delta: Change in game score (score_t - score_{t-1}).
            novelty: Novelty score from the memory graph (0–1).
            prediction_error: World model's prediction error (curiosity).
            **kwargs: Additional signals that could be incorporated.

        Returns:
            Scalar utility value.
        """
        # Clip curiosity to prevent runaway values
        curiosity = min(prediction_error, self.config.curiosity_clip)

        # Normalize extrinsic by running baseline (helps across games
        # with different score scales)
        normalized_extrinsic = extrinsic_delta
        if self._running_baseline > 0:
            normalized_extrinsic = extrinsic_delta / max(abs(self._running_baseline), 1.0)

        # Update running baseline (exponential moving average)
        if extrinsic_delta != 0:
            self._running_baseline = (
                0.99 * self._running_baseline + 0.01 * abs(extrinsic_delta)
            )
        self._update_count += 1

        utility = (
            self.config.w_extrinsic * normalized_extrinsic
            + self.config.w_novelty * novelty
            + self.config.w_curiosity * curiosity
        )

        return float(utility)

    def compute_for_latent(
        self,
        novelty: float,
        prediction_error: float,
        score_delta: float = 0.0,
    ) -> float:
        """Compute utility for a latent state (used by the planner).

        Args:
            novelty: Novelty of the state.
            prediction_error: Model's prediction error for reaching this state.
            score_delta: Score change when entering this state.

        Returns:
            Utility value.
        """
        return self.compute(
            extrinsic_delta=score_delta,
            novelty=novelty,
            prediction_error=prediction_error,
        )

    def update_weights(self, **new_weights: float) -> None:
        """Dynamically update reward weights (e.g. via meta-learning).

        Args:
            **new_weights: Keyword arguments matching RewardConfig fields.
        """
        if "w_extrinsic" in new_weights:
            self.config.w_extrinsic = new_weights["w_extrinsic"]
        if "w_novelty" in new_weights:
            self.config.w_novelty = new_weights["w_novelty"]
        if "w_curiosity" in new_weights:
            self.config.w_curiosity = new_weights["w_curiosity"]

        logger.debug("Updated reward weights: %s", new_weights)

    def reset(self) -> None:
        """Reset per-level state (running baseline)."""
        self._running_baseline = self.config.score_baseline
        self._update_count = 0
