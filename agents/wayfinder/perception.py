"""Perception encoder — Module A.

Converts a raw 64×64 grid of 4-bit color indices (0–15) into a compact
latent representation using a small from-scratch CNN.

The encoder also computes a binary diff mask against the previous frame,
which is used by the world model to predict frame changes.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

GRID_SIZE = 64
NUM_COLORS = 16
LATENT_DIM_DEFAULT = 256


class PerceptionEncoder(nn.Module):
    """CNN encoder for 64×64×16 one-hot frames.

    Architecture (3 conv layers + 1 FC head):
        Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool2d(2)  # 64→32
        Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)  # 32→16
        Conv2d(64, 128, 3, padding=1) → ReLU → MaxPool2d(2) # 16→8
        Flatten → Linear(128*8*8, latent_dim)

    No pretrained weights exist for this domain — trained from scratch.

    Attributes:
        latent_dim: Output latent vector dimensionality.
        device: Torch device for inference.
    """

    def __init__(self, latent_dim: int = LATENT_DIM_DEFAULT, device: str = "cpu") -> None:
        """Initialize the encoder.

        Args:
            latent_dim: Output latent dimension.
            device: Torch device ("cpu" or "cuda").
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 → 8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

        self.to(self.device)
        self.eval()
        logger.info("PerceptionEncoder initialized (latent_dim=%d, device=%s)", latent_dim, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: one-hot frame → latent vector.

        Args:
            x: Tensor of shape (B, 16, 64, 64) — one-hot encoded frames.

        Returns:
            Latent tensor of shape (B, latent_dim).
        """
        features = self.conv(x)
        latent = self.fc(features)
        return latent

    def encode(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode a single frame into a latent vector + diff mask.

        Args:
            frame: 64×64 numpy array of uint8 color indices (0–15).

        Returns:
            Tuple of (latent, diff_mask):
            - latent: float32 array of shape (latent_dim,).
            - diff_mask: 64×64 bool array — True where frame differs from previous.
        """
        one_hot = self._to_one_hot(frame)
        with torch.no_grad():
            tensor = torch.from_numpy(one_hot).unsqueeze(0).to(self.device)
            latent = self.forward(tensor).squeeze(0).cpu().numpy()

        # Compute diff mask against previous frame
        diff_mask = self._compute_diff(frame)

        # Store current frame for next step's diff
        self._prev_frame = frame.copy()

        return latent.astype(np.float32), diff_mask

    def _to_one_hot(self, frame: np.ndarray) -> np.ndarray:
        """Convert a 64×64 integer frame to 16×64×64 one-hot float.

        Args:
            frame: 64×64 uint8 array with values in [0, 15].

        Returns:
            16×64×64 float32 one-hot array.
        """
        one_hot = np.zeros((NUM_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for c in range(NUM_COLORS):
            one_hot[c] = (frame == c).astype(np.float32)
        return one_hot

    def _compute_diff(self, frame: np.ndarray) -> np.ndarray:
        """Compute binary diff mask against the previous frame.

        Args:
            frame: Current 64×64 frame.

        Returns:
            64×64 bool array — True where pixels changed.
        """
        if not hasattr(self, "_prev_frame"):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        return frame != self._prev_frame

    def encode_batch(self, frames: np.ndarray) -> np.ndarray:
        """Encode a batch of frames into latent vectors.

        Used during offline training of the world model.

        Args:
            frames: (B, 64, 64) uint8 array.

        Returns:
            (B, latent_dim) float32 array.
        """
        one_hot = np.zeros((len(frames), NUM_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, frame in enumerate(frames):
            for c in range(NUM_COLORS):
                one_hot[i, c] = (frame == c).astype(np.float32)

        with torch.no_grad():
            tensor = torch.from_numpy(one_hot).to(self.device)
            latents = self.forward(tensor).cpu().numpy()

        return latents.astype(np.float32)
