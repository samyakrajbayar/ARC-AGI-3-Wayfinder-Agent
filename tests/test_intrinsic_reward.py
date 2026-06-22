"""Tests for the intrinsic reward module."""

from __future__ import annotations

import pytest

from agents.wayfinder.intrinsic_reward import IntrinsicReward, RewardConfig


class TestIntrinsicReward:
    """Test cases for IntrinsicReward."""

    @pytest.fixture
    def reward(self) -> IntrinsicReward:
        """Create a test intrinsic reward module."""
        return IntrinsicReward(RewardConfig(w_extrinsic=1.0, w_novelty=0.3, w_curiosity=0.2))

    def test_compute_returns_float(self, reward: IntrinsicReward) -> None:
        """Test that compute returns a float."""
        result = reward.compute(extrinsic_delta=1.0, novelty=0.5, prediction_error=2.0)
        assert isinstance(result, float)

    def test_higher_extrinsic_gives_higher_utility(self, reward: IntrinsicReward) -> None:
        """Test that higher extrinsic delta increases utility."""
        low = reward.compute(extrinsic_delta=0.0, novelty=0.5, prediction_error=1.0)
        high = reward.compute(extrinsic_delta=1.0, novelty=0.5, prediction_error=1.0)
        assert high > low

    def test_higher_novelty_gives_higher_utility(self, reward: IntrinsicReward) -> None:
        """Test that higher novelty increases utility."""
        low = reward.compute(extrinsic_delta=0.5, novelty=0.0, prediction_error=1.0)
        high = reward.compute(extrinsic_delta=0.5, novelty=1.0, prediction_error=1.0)
        assert high > low

    def test_curiosity_clip(self) -> None:
        """Test that curiosity is clipped."""
        reward = IntrinsicReward(RewardConfig(w_curiosity=1.0, curiosity_clip=5.0))
        clipped = reward.compute(extrinsic_delta=0.0, novelty=0.0, prediction_error=100.0)
        # Curiosity is clipped to 5.0, and w_curiosity=1.0, so utility should be 5.0
        assert abs(clipped - 5.0) < 0.1

    def test_update_weights(self, reward: IntrinsicReward) -> None:
        """Test that weights can be updated."""
        reward.update_weights(w_extrinsic=2.0)
        assert reward.config.w_extrinsic == 2.0

    def test_reset(self, reward: IntrinsicReward) -> None:
        """Test that reset clears running baseline."""
        reward.compute(extrinsic_delta=10.0)
        reward.reset()
        assert reward._running_baseline == 0.0

    def test_config_from_yaml(self, tmp_path) -> None:
        """Test loading config from YAML."""
        import yaml
        config_data = {"intrinsic_reward": {"w_extrinsic": 2.0, "w_novelty": 0.5}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = RewardConfig.from_yaml(config_path)
        assert config.w_extrinsic == 2.0
        assert config.w_novelty == 0.5
