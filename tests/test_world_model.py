"""Tests for the world model module."""

from __future__ import annotations

import numpy as np
import pytest

from agents.wayfinder.world_model import WorldModel, Transition


class TestWorldModel:
    """Test cases for WorldModel."""

    @pytest.fixture
    def model(self) -> WorldModel:
        """Create a test world model."""
        return WorldModel(latent_dim=32, buffer_size=1000, device="cpu")

    @pytest.fixture
    def sample_latent(self) -> np.ndarray:
        """Create a sample latent vector."""
        return np.random.randn(32).astype(np.float32)

    def test_predict_change_returns_probability(self, model: WorldModel, sample_latent: np.ndarray) -> None:
        """Test that predict_change returns a valid probability."""
        prob = model.predict_change(sample_latent, {"action": "ACTION1"})
        assert 0.0 <= prob <= 1.0

    def test_predict_next_latent_shape(self, model: WorldModel, sample_latent: np.ndarray) -> None:
        """Test that predict_next_latent returns correct shape."""
        next_latent = model.predict_next_latent(sample_latent, {"action": "ACTION1"})
        assert next_latent.shape == (32,)

    def test_add_transition_increases_buffer(self, model: WorldModel, sample_latent: np.ndarray) -> None:
        """Test that add_transition adds to the buffer."""
        assert model.buffer_size_current == 0
        model.add_transition(
            state_latent=sample_latent,
            action={"action": "ACTION1"},
            next_latent=sample_latent + 0.1,
            frame_changed=True,
        )
        assert model.buffer_size_current == 1

    def test_add_transition_deduplicates(self, model: WorldModel, sample_latent: np.ndarray) -> None:
        """Test that identical transitions are deduplicated."""
        for _ in range(3):
            model.add_transition(
                state_latent=sample_latent,
                action={"action": "ACTION1"},
                next_latent=sample_latent + 0.1,
                frame_changed=True,
            )
        assert model.buffer_size_current == 1

    def test_train_step_returns_loss(self, model: WorldModel, sample_latent: np.ndarray) -> None:
        """Test that train_step returns a loss value after enough data."""
        # Add enough transitions for a batch
        for i in range(70):
            model.add_transition(
                state_latent=sample_latent + i * 0.01,
                action={"action": f"ACTION{i % 4 + 1}"},
                next_latent=sample_latent + (i + 1) * 0.01,
                frame_changed=(i % 2 == 0),
            )
        loss = model.train_step(batch_size=32)
        assert loss >= 0.0

    def test_confidence_starts_low(self, model: WorldModel) -> None:
        """Test that confidence is 0 before training."""
        assert model.confidence() == 0.0

    def test_prediction_error_zero_on_first_step(self, model: WorldModel) -> None:
        """Test that prediction error is 0 on the first step."""
        latent = np.random.randn(32).astype(np.float32)
        assert model.prediction_error(None, None, latent) == 0.0
