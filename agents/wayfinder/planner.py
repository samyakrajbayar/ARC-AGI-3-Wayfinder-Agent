"""Planner — Module E.

Short-horizon tree search using the world model as a cheap simulator
and the intrinsic reward module as the value function.

Adapts the rollout/backup structure from the Topdeck ISMCTS agent,
simplified since ARC-AGI-3 frames are fully observed (no information-set
handling needed). The search budget scales with the remaining action
allowance — more budget early in a level, less when running low.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

from agents.wayfinder.world_model import WorldModel

logger = logging.getLogger(__name__)

# Type alias for the utility function
UtilityFn = Callable[..., float]


class PlannerNode:
    """A node in the search tree.

    Attributes:
        latent: State latent vector at this node.
        action: Action that led to this node (None for root).
        parent: Parent node (None for root).
        children: List of child PlannerNodes.
        visits: Number of times this node has been visited in search.
        total_value: Accumulated utility from rollouts through this node.
        is_terminal: Whether this is a terminal state.
    """

    __slots__ = ("latent", "action", "action_data", "parent", "children",
                 "visits", "total_value", "is_terminal")

    def __init__(
        self,
        latent: np.ndarray,
        action: str | None = None,
        action_data: dict | None = None,
        parent: "PlannerNode | None" = None,
        is_terminal: bool = False,
    ) -> None:
        self.latent = latent
        self.action = action
        self.action_data = action_data
        self.parent = parent
        self.children: list[PlannerNode] = []
        self.visits = 0
        self.total_value = 0.0
        self.is_terminal = is_terminal

    @property
    def mean_value(self) -> float:
        """Mean utility value from rollouts through this node."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, exploration: float = 1.414) -> float:
        """UCB1 selection score.

        Args:
            exploration: Exploration constant (sqrt(2) by default).

        Returns:
            UCB1 value for node selection.
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploit = self.mean_value
        explore = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore

    def best_child(self, exploration: float = 1.414) -> "PlannerNode | None":
        """Select the child with the highest UCB1 score.

        Args:
            exploration: UCB1 exploration constant.

        Returns:
            Best child node, or None if no children.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1(exploration))


class Planner:
    """Short-horizon tree search planner.

    Uses the world model to simulate forward, the intrinsic reward to
    evaluate states, and UCB1-based selection (like MCTS/ISMCTS) to
    balance exploration and exploitation in the search tree.

    The search depth and number of simulations scale with the remaining
    action budget — more budget means deeper search.

    Attributes:
        world_model: The world/transition model (used as simulator).
        max_depth: Maximum search depth per simulation.
        max_simulations: Number of MCTS simulations per planning step.
    """

    ALL_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]

    def __init__(
        self,
        world_model: WorldModel,
        reward_module: Any,
        max_depth: int = 5,
        max_simulations: int = 50,
    ) -> None:
        """Initialize the planner.

        Args:
            world_model: World model for forward simulation.
            reward_module: Intrinsic reward module for value estimation.
            max_depth: Max rollout depth.
            max_simulations: Number of MCTS simulations per plan() call.
        """
        self.world_model = world_model
        self.reward_module = reward_module
        self.max_depth = max_depth
        self.max_simulations = max_simulations

        logger.info(
            "Planner initialized (max_depth=%d, max_sims=%d)",
            max_depth,
            max_simulations,
        )

    def plan(
        self,
        latent: np.ndarray,
        available_actions: list[str],
        action_budget_remaining: int,
        utility_fn: UtilityFn,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plan the next action using tree search.

        Args:
            latent: Current state latent.
            available_actions: Actions available this step.
            action_budget_remaining: Remaining action budget.
            utility_fn: Function to compute utility of a state.
            **kwargs: Additional context.

        Returns:
            Action dict with "action" and optional "data" keys.
        """
        # Scale simulations with remaining budget
        budget_factor = min(1.0, action_budget_remaining / 100.0)
        num_sims = max(5, int(self.max_simulations * budget_factor))
        depth = max(2, int(self.max_depth * budget_factor))

        # Filter to simple actions for tree search (ACTION6's 4096-position
        # space is too large for full tree search — handled by action head)
        search_actions = [a for a in available_actions if a in self.ALL_ACTIONS]
        if not search_actions:
            search_actions = self.ALL_ACTIONS[:3]

        root = PlannerNode(latent=latent)

        # Run simulations
        for _ in range(num_sims):
            self._simulate(root, search_actions, depth, utility_fn)

        # Select the best action from root's children
        best = root.best_child(exploration=0.0)  # Greedy at root
        if best is None or best.action is None:
            # Fallback: pick first available action
            return {"action": available_actions[0]}

        result: dict[str, Any] = {"action": best.action}
        if best.action_data:
            result["data"] = best.action_data
        return result

    def _simulate(
        self,
        root: PlannerNode,
        actions: list[str],
        max_depth: int,
        utility_fn: UtilityFn,
    ) -> float:
        """Run one MCTS simulation: select → expand → rollout → backup.

        Args:
            root: Root of the search tree.
            actions: Available actions for expansion.
            max_depth: Maximum rollout depth.
            utility_fn: Utility function for leaf evaluation.

        Returns:
            Accumulated utility from the rollout.
        """
        # --- Selection ---
        node = root
        depth = 0
        while node.children and not node.is_terminal and depth < max_depth:
            node = node.best_child()
            if node is None:
                break
            depth += 1

        # --- Expansion ---
        if node is not None and not node.is_terminal and depth < max_depth:
            # Expand: add children for all actions
            for action in actions:
                action_dict = {"action": action}
                # Use world model to predict next latent
                next_latent = self.world_model.predict_next_latent(
                    node.latent, action_dict
                )
                child = PlannerNode(
                    latent=next_latent,
                    action=action,
                    parent=node,
                )
                node.children.append(child)
            # Pick a child for rollout
            if node.children:
                node = np.random.choice(node.children)

        # --- Rollout (random forward simulation) ---
        if node is not None:
            total_utility = self._rollout(
                node.latent, actions, max_depth - depth, utility_fn
            )
        else:
            total_utility = 0.0

        # --- Backup ---
        while node is not None:
            node.visits += 1
            node.total_value += total_utility
            node = node.parent

        return total_utility

    def _rollout(
        self,
        latent: np.ndarray,
        actions: list[str],
        depth: int,
        utility_fn: UtilityFn,
    ) -> float:
        """Random rollout from a latent state.

        Args:
            latent: Starting latent.
            actions: Available actions.
            depth: Remaining rollout depth.
            utility_fn: Utility function.

        Returns:
            Accumulated utility.
        """
        total = 0.0
        current = latent

        for _ in range(max(depth, 1)):
            action = np.random.choice(actions)
            action_dict = {"action": action}

            # Simulate forward
            next_latent = self.world_model.predict_next_latent(current, action_dict)

            # Estimate utility (novelty ≈ 0 in simulation since we're
            # predicting latents, not real frames; use prediction error
            # as a proxy for curiosity)
            pred_error = float(np.linalg.norm(next_latent - current))
            utility = utility_fn(
                extrinsic_delta=0.0,
                novelty=0.0,
                prediction_error=pred_error,
            )
            total += utility * (0.95 ** _)  # Discount
            current = next_latent

        return total
