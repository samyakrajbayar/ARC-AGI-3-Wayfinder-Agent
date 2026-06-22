"""Tests for the perception encoder module."""

from __future__ import annotations

import numpy as np
import pytest

from agents.wayfinder.perception import PerceptionEncoder, GRID_SIZE, NUM_COLORS


class TestPerceptionEncoder:
    """Test cases for PerceptionEncoder."""

    @pytest.fixture
    def encoder(self) -> PerceptionEncoder:
        """Create a test encoder instance."""
        return PerceptionEncoder(latent_dim=64, device="cpu")

    @pytest.fixture
    def sample_frame(self) -> np.ndarray:
        """Create a sample 64×64 frame."""
        return np.random.randint(0, NUM_COLORS, size=(GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    def test_encode_returns_correct_shapes(self, encoder: PerceptionEncoder, sample_frame: np.ndarray) -> None:
        """Test that encode returns latent and diff_mask with correct shapes."""
        latent, diff_mask = encoder.encode(sample_frame)
        assert latent.shape == (64,)
        assert diff_mask.shape == (GRID_SIZE, GRID_SIZE)
        assert diff_mask.dtype == bool

    def test_first_frame_diff_is_zero(self, encoder: PerceptionEncoder, sample_frame: np.ndarray) -> None:
        """Test that the first frame has an all-zero diff mask."""
        _, diff_mask = encoder.encode(sample_frame)
        assert not np.any(diff_mask)

    def test_second_frame_diff_detects_changes(self, encoder: PerceptionEncoder, sample_frame: np.ndarray) -> None:
        """Test that diff mask detects changes between frames."""
        encoder.encode(sample_frame)
        modified = sample_frame.copy()
        modified[0, 0] = (modified[0, 0] + 1) % NUM_COLORS
        _, diff_mask = encoder.encode(modified)
        assert diff_mask[0, 0]
        assert not diff_mask[1, 1]  # Unchanged pixel

    def test_encode_batch(self, encoder: PerceptionEncoder) -> None:
        """Test batch encoding."""
        frames = np.random.randint(0, NUM_COLORS, size=(4, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        latents = encoder.encode_batch(frames)
        assert latents.shape == (4, 64)

    def test_latent_is_float32(self, encoder: PerceptionEncoder, sample_frame: np.ndarray) -> None:
        """Test that the latent is float32."""
        latent, _ = encoder.encode(sample_frame)
        assert latent.dtype == np.float32
