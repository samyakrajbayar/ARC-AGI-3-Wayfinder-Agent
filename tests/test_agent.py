"""Integration tests for the WayfinderAgent."""

from __future__ import annotations

import numpy as np
import pytest

from agents.wayfinder.agent import WayfinderAgent


class TestWayfinderAgent:
    """Integration test cases for WayfinderAgent."""

    @pytest.fixture
    def agent(self) -> WayfinderAgent:
        """Create a test agent with small dimensions."""
        return WayfinderAgent(
            max_actions=100,
            latent_dim=32,
            buffer_size=1000,
            device="cpu",
        )

    @pytest.fixture
    def frame(self) -> np.ndarray:
        """Create a sample frame."""
        return np.random.randint(0, 16, size=(64, 64), dtype=np.uint8)

    def test_act_returns_valid_action(self, agent: WayfinderAgent, frame: np.ndarray) -> None:
        """Test that act returns a valid action dict."""
        result = agent.act(
            frames=[frame],
            state="NOT_FINISHED",
            score=0.0,
            win_score=1.0,
            available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
        )
        assert "action" in result
        assert "reasoning" in result
        assert result["action"] in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]

    def test_is_done_on_win(self, agent: WayfinderAgent) -> None:
        """Test that is_done returns True on WIN state."""
        assert agent.is_done([], "WIN")

    def test_is_done_on_game_over(self, agent: WayfinderAgent) -> None:
        """Test that is_done returns True on GAME_OVER state."""
        assert agent.is_done([], "GAME_OVER")

    def test_is_done_on_budget_exhausted(self, agent: WayfinderAgent) -> None:
        """Test that is_done returns True when action budget is exhausted."""
        agent.action_count = agent.max_actions
        assert agent.is_done([], "NOT_FINISHED")

    def test_is_done_false_during_play(self, agent: WayfinderAgent) -> None:
        """Test that is_done returns False during normal play."""
        agent.action_count = 5
        assert not agent.is_done([], "NOT_FINISHED")

    def test_reset_clears_state(self, agent: WayfinderAgent, frame: np.ndarray) -> None:
        """Test that reset clears per-level state."""
        agent.act(
            frames=[frame],
            state="NOT_FINISHED",
            score=0.0,
            win_score=1.0,
            available_actions=["ACTION1"],
        )
        assert agent.action_count > 0

        agent.reset()
        assert agent.action_count == 0
        assert agent._prev_latent is None

    def test_multiple_steps_accumulate_transitions(
        self, agent: WayfinderAgent, frame: np.ndarray
    ) -> None:
        """Test that multiple steps build up the world model buffer."""
        for i in range(5):
            f = np.random.randint(0, 16, size=(64, 64), dtype=np.uint8)
            agent.act(
                frames=[f],
                state="NOT_FINISHED",
                score=0.0,
                win_score=1.0,
                available_actions=["ACTION1", "ACTION2", "ACTION3"],
            )
        # After 5 steps, at least 3 transitions should be in the buffer
        assert agent._world_model.buffer_size_current >= 3

    def test_reasoning_blob_has_required_fields(self, agent: WayfinderAgent, frame: np.ndarray) -> None:
        """Test that the reasoning blob contains required audit fields."""
        result = agent.act(
            frames=[frame],
            state="NOT_FINISHED",
            score=0.0,
            win_score=1.0,
            available_actions=["ACTION1"],
        )
        reasoning = result["reasoning"]
        assert "step" in reasoning
        assert "score" in reasoning
        assert "model_confidence" in reasoning
        assert "novelty" in reasoning
