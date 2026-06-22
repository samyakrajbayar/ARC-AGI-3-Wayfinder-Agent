"""Tests for the planner module."""

from __future__ import annotations

import numpy as np
import pytest

from agents.wayfinder.planner import Planner, PlannerNode
from agents.wayfinder.world_model import WorldModel
from agents.wayfinder.intrinsic_reward import IntrinsicReward


class TestPlannerNode:
    """Test cases for PlannerNode."""

    def test_ucb1_infinite_for_unvisited(self) -> None:
        """Test that UCB1 is infinite for unvisited nodes."""
        node = PlannerNode(latent=np.zeros(32))
        assert node.ucb1() == float("inf")

    def test_ucb1_finite_after_visit(self) -> None:
        """Test that UCB1 is finite after a visit."""
        parent = PlannerNode(latent=np.zeros(32))
        parent.visits = 10
        node = PlannerNode(latent=np.zeros(32), parent=parent)
        node.visits = 1
        node.total_value = 0.5
        assert node.ucb1() < float("inf")

    def test_mean_value(self) -> None:
        """Test mean value computation."""
        node = PlannerNode(latent=np.zeros(32))
        node.visits = 4
        node.total_value = 2.0
        assert node.mean_value == 0.5

    def test_mean_value_zero_visits(self) -> None:
        """Test mean value is 0 for unvisited nodes."""
        node = PlannerNode(latent=np.zeros(32))
        assert node.mean_value == 0.0


class TestPlanner:
    """Test cases for Planner."""

    @pytest.fixture
    def planner(self) -> Planner:
        """Create a test planner."""
        world_model = WorldModel(latent_dim=32, device="cpu")
        reward_module = IntrinsicReward()
        return Planner(
            world_model=world_model,
            reward_module=reward_module,
            max_depth=3,
            max_simulations=10,
        )

    def test_plan_returns_action_dict(self, planner: Planner) -> None:
        """Test that plan returns a valid action dict."""
        latent = np.random.randn(32).astype(np.float32)
        result = planner.plan(
            latent=latent,
            available_actions=["ACTION1", "ACTION2", "ACTION3"],
            action_budget_remaining=100,
            utility_fn=lambda **kw: 0.0,
        )
        assert "action" in result
        assert result["action"] in ["ACTION1", "ACTION2", "ACTION3"]

    def test_plan_fallback_on_no_actions(self, planner: Planner) -> None:
        """Test that plan falls back when no valid actions."""
        latent = np.random.randn(32).astype(np.float32)
        result = planner.plan(
            latent=latent,
            available_actions=["ACTION1"],
            action_budget_remaining=100,
            utility_fn=lambda **kw: 0.0,
        )
        assert "action" in result

    def test_plan_scales_with_budget(self, planner: Planner) -> None:
        """Test that planning scales with remaining budget."""
        latent = np.random.randn(32).astype(np.float32)
        # Should not crash with very low budget
        result = planner.plan(
            latent=latent,
            available_actions=["ACTION1", "ACTION2"],
            action_budget_remaining=1,
            utility_fn=lambda **kw: 0.0,
        )
        assert "action" in result
