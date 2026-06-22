"""Action head — Module F.

Hierarchical action selection: first choose an action type (RESET,
ACTION1–ACTION5, ACTION6), then for ACTION6 choose a click coordinate
via a conv-based coordinate head producing a 64×64 heatmap.

The action type is selected via softmax over per-type change-probabilities
(from the world model) with epsilon-greedy exploration. The coordinate
head uses a small CNN to produce a spatial heatmap over the 64×64 grid.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.wayfinder.world_model import WorldModel

logger = logging.getLogger(__name__)

GRID_SIZE = 64
NUM_COLORS = 16


class CoordinateHead(nn.Module):
    """Conv-based coordinate prediction head for ACTION6.

    Takes the one-hot frame + latent as input and produces a 64×64
    heatmap over click targets. This is more sample-efficient than a
    flat 4096-way softmax because it exploits spatial structure.

    Architecture:
        Input: (one_hot[16, 64, 64] + latent_tile[1, 64, 64]) → 17 channels
        Conv2d(17, 32, 3, padding=1) → ReLU
        Conv2d(32, 16, 3, padding=1) → ReLU
        Conv2d(16, 1, 1) → Sigmoid → 64×64 heatmap
    """

    def __init__(self, latent_dim: int = 256) -> None:
        """Initialize the coordinate head.

        Args:
            latent_dim: Dimension of the input latent vector.
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, one_hot: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            one_hot: (B, 16, 64, 64) one-hot frame.
            latent: (B, latent_dim) latent vector.

        Returns:
            (B, 64, 64) heatmap with values in [0, 1].
        """
        # Broadcast latent to spatial dimensions
        latent_map = latent[:, :1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        latent_map = latent_map.expand(-1, -1, GRID_SIZE, GRID_SIZE)  # (B, 1, 64, 64)

        x = torch.cat([one_hot, latent_map], dim=1)  # (B, 17, 64, 64)
        heatmap = self.conv(x).squeeze(1)  # (B, 64, 64)
        return torch.sigmoid(heatmap)


class ActionHead:
    """Hierarchical action selection head.

    Phase 1: Choose action type via softmax over per-type change-probabilities.
    Phase 2: For ACTION6, use the coordinate head to pick a click target.

    Attributes:
        latent_dim: Input latent dimension.
        device: Torch device.
        coord_head: Coordinate prediction CNN.
    """

    def __init__(self, latent_dim: int = 256, device: str = "cpu") -> None:
        """Initialize the action head.

        Args:
            latent_dim: Latent dimension from the perception encoder.
            device: Torch device.
        """
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.coord_head = CoordinateHead(latent_dim=latent_dim).to(self.device)
        self.coord_head.eval()

        logger.info("ActionHead initialized (latent_dim=%d, device=%s)", latent_dim, device)

    def select(
        self,
        latent: np.ndarray,
        diff_mask: np.ndarray,
        available_actions: list[str],
        world_model: WorldModel,
        epsilon: float = 0.1,
    ) -> dict[str, object]:
        """Select an action hierarchically.

        Args:
            latent: Current state latent vector.
            diff_mask: 64×64 boolean diff mask (where frame changed).
            available_actions: Actions available this step.
            world_model: World model for change prediction.
            epsilon: Exploration probability for epsilon-greedy.

        Returns:
            Action dict with "action" (str) and optional "data" (dict).
        """
        # --- Epsilon-greedy: random exploration ---
        if np.random.random() < epsilon:
            action_name = str(np.random.choice(available_actions))
            if action_name == "ACTION6":
                return self._select_coordinate(latent)
            return {"action": action_name}

        # --- Phase 1: Choose action type ---
        # Compute P(frame changes) for each available action
        change_probs = []
        for action_name in available_actions:
            if action_name == "RESET":
                change_probs.append(0.1)  # RESET always changes the frame
                continue
            prob = world_model.predict_change(latent, {"action": action_name})
            change_probs.append(prob)

        change_probs_arr = np.array(change_probs, dtype=np.float32)

        # Softmax with temperature
        temperature = 0.5
        logits = np.log(change_probs_arr + 1e-7) / temperature
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()

        selected_idx = np.random.choice(len(available_actions), p=probs)
        action_name = available_actions[selected_idx]

        # --- Phase 2: For ACTION6, select coordinates ---
        if action_name == "ACTION6":
            return self._select_coordinate(latent)

        return {"action": action_name}

    def _select_coordinate(self, latent: np.ndarray) -> dict[str, object]:
        """Select a click coordinate for ACTION6.

        Uses the coordinate head to produce a 64×64 heatmap, then samples
        from it as a categorical distribution. Falls back to uniform random
        if the heatmap is degenerate.

        Args:
            latent: Current state latent vector.

        Returns:
            Action dict with "action": "ACTION6" and "data": {"x": int, "y": int}.
        """
        # Build a dummy one-hot from the latent (in practice, we'd use the
        # actual frame; here we create a placeholder since the action head
        # receives latents, not raw frames — in the full implementation,
        # we'd pass the frame through or cache it)
        one_hot = torch.zeros(1, NUM_COLORS, GRID_SIZE, GRID_SIZE, device=self.device)
        # Place the latent-derived signal in channel 0 as a spatial broadcast
        lat_tensor = torch.from_numpy(latent).unsqueeze(0).to(self.device)
        one_hot[:, 0] = lat_tensor[:, :1].unsqueeze(-1).expand(-1, -1, GRID_SIZE).squeeze(0).unsqueeze(0)

        with torch.no_grad():
            heatmap = self.coord_head(one_hot, lat_tensor)  # (1, 64, 64)
            heatmap = heatmap.squeeze(0).cpu().numpy()  # (64, 64)

        # Flatten and sample
        flat = heatmap.flatten()
        flat = flat - flat.min()
        total = flat.sum()

        if total < 1e-7:
            # Degenerate heatmap — uniform random
            x = int(np.random.randint(0, GRID_SIZE))
            y = int(np.random.randint(0, GRID_SIZE))
        else:
            probs = flat / total
            idx = np.random.choice(GRID_SIZE * GRID_SIZE, p=probs)
            y, x = divmod(idx, GRID_SIZE)

        return {"action": "ACTION6", "data": {"x": int(x), "y": int(y)}}

    def train_coordinate_head(
        self,
        frames: np.ndarray,
        latents: np.ndarray,
        targets: np.ndarray,
        lr: float = 1e-3,
        epochs: int = 10,
    ) -> float:
        """Train the coordinate head on labeled click targets.

        Used during offline training when click data is available.

        Args:
            frames: (B, 64, 64) uint8 frames.
            latents: (B, latent_dim) latent vectors.
            targets: (B, 2) array of (x, y) click coordinates.
            lr: Learning rate.
            epochs: Number of training epochs.

        Returns:
            Final training loss.
        """
        self.coord_head.train()
        optimizer = torch.optim.Adam(self.coord_head.parameters(), lr=lr)

        # Convert frames to one-hot
        one_hot = np.zeros((len(frames), NUM_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i, frame in enumerate(frames):
            for c in range(NUM_COLORS):
                one_hot[i, c] = (frame == c).astype(np.float32)

        one_hot_t = torch.from_numpy(one_hot).to(self.device)
        latents_t = torch.from_numpy(latents.astype(np.float32)).to(self.device)

        # Create target heatmaps (Gaussian around target)
        target_maps = torch.zeros(len(frames), GRID_SIZE, GRID_SIZE, device=self.device)
        for i, (x, y) in enumerate(targets):
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = int(y) + dy, int(x) + dx
                    if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                        target_maps[i, ny, nx] = np.exp(-(dx**2 + dy**2) / 2.0)

        loss_fn = nn.BCELoss()
        final_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.coord_head(one_hot_t, latents_t)
            loss = loss_fn(pred, target_maps)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.coord_head.eval()
        return final_loss
