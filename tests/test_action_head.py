"""Tests for the action head module."""

from __future__ import annotations

import numpy as np
import pytest

from agents.wayfinder.action_head import ActionHead
from agents.wayfinder.world_model import WorldModel


class TestActionHead:
    """Test cases for ActionHead."""

    @pytest.fixture
    def action_head(self) -> ActionHead:
        """Create a test action head."""
        return ActionHead(latent_dim=32, device="cpu")

    @pytest.fixture
    def world_model(self) -> WorldModel:
        """Create a test world model."""
        return WorldModel(latent_dim=32, device="cpu")

    @pytest.fixture
    def latent(self) -> np.ndarray:
        """Create a sample latent."""
        return np.random.randn(32).astype(np.float32)

    def test_select_returns_valid_action(
        self, action_head: ActionHead, world_model: WorldModel, latent: np.ndarray
    ) -> None:
        """Test that select returns a valid action."""
        result = action_head.select(
            latent=latent,
            diff_mask=np.zeros((64, 64), dtype=bool),
            available_actions=["ACTION1", "ACTION2", "ACTION3"],
            world_model=world_model,
            epsilon=0.0,
        )
        assert "action" in result
        assert result["action"] in ["ACTION1", "ACTION2", "ACTION3"]

    def test_select_action6_includes_coordinates(
        self, action_head: ActionHead, world_model: WorldModel, latent: np.ndarray
    ) -> None:
        """Test that ACTION6 includes x, y coordinates."""
        result = action_head.select(
            latent=latent,
            diff_mask=np.zeros((64, 64), dtype=bool),
            available_actions=["ACTION6"],
            world_model=world_model,
            epsilon=0.0,
        )
        assert result["action"] == "ACTION6"
        assert "data" in result
        assert "x" in result["data"]
        assert "y" in result["data"]
        assert 0 <= result["data"]["x"] < 64
        assert 0 <= result["data"]["y"] < 64

    def test_epsilon_greedy_can_explore(
        self, action_head: ActionHead, world_model: WorldModel, latent: np.ndarray
    ) -> None:
        """Test that epsilon > 0 allows random exploration."""
        actions_taken: set[str] = set()
        for _ in range(50):
            result = action_head.select(
                latent=latent,
                diff_mask=np.zeros((64, 64), dtype=bool),
                available_actions=["ACTION1", "ACTION2", "ACTION3"],
                world_model=world_model,
                epsilon=1.0,  # Always explore
            )
            actions_taken.add(result["action"])
        # Should have tried multiple different actions
        assert len(actions_taken) > 1
