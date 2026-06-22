"""World/transition model — Module B.

A self-supervised model that learns two things from played transitions:
1. P(frame changes | state, action) — a binary change-prediction head.
2. A forward model ŝ_{t+1} = f(s_t, a_t) predicting the next latent.

Trained online from a deduplicated replay buffer. The change-prediction
head is the primary signal for the reactive policy (like StochasticGoose),
while the forward model enables the planner's tree search.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single (state, action, next_state, changed) transition.

    Attributes:
        state_latent: Latent vector of the state before the action.
        action: Action name string (e.g. "ACTION1", "ACTION6").
        action_data: Optional action data (e.g. {"x": 32, "y": 32} for ACTION6).
        next_latent: Latent vector of the state after the action.
        frame_changed: Whether the frame changed as a result of the action.
    """

    state_latent: np.ndarray
    action: str
    action_data: dict | None
    next_latent: np.ndarray
    frame_changed: bool


class WorldModel(nn.Module):
    """World/transition model for ARC-AGI-3.

    Architecture:
    - Action embedding: 7 action types → 32-dim embedding.
    - Change predictor: MLP(latent_dim + action_emb) → 1 (sigmoid).
    - Forward model: MLP(latent_dim + action_emb) → latent_dim.

    The change predictor is trained with BCE loss + entropy regularization.
    The forward model is trained with MSE loss on the latent difference.

    Attributes:
        latent_dim: Dimension of the perception encoder's output.
        buffer_size: Max transitions in the replay buffer.
        device: Torch device.
    """

    ACTION_TYPES = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
    ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}

    def __init__(
        self,
        latent_dim: int = 256,
        buffer_size: int = 200_000,
        device: str = "cpu",
        lr: float = 1e-4,
    ) -> None:
        """Initialize the world model.

        Args:
            latent_dim: Input latent dimensionality.
            buffer_size: Maximum transitions stored.
            device: Torch device.
            lr: Learning rate for the optimizer.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.lr = lr

        # Action embedding
        self.action_embed = nn.Embedding(len(self.ACTION_TYPES), 32)

        # Change prediction head: P(frame changes | state, action)
        self.change_head = nn.Sequential(
            nn.Linear(latent_dim + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Forward model: predict next latent
        self.forward_head = nn.Sequential(
            nn.Linear(latent_dim + 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

        # Replay buffer (deduplicated by state hash)
        self._buffer: deque[Transition] = deque(maxlen=buffer_size)
        self._seen_hashes: set[str] = set()

        # Training stats
        self._train_steps = 0
        self._change_loss_avg = 0.0
        self._forward_loss_avg = 0.0

        logger.info("WorldModel initialized (latent_dim=%d, buffer=%d)", latent_dim, buffer_size)

    def forward(self, latent: torch.Tensor, action_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            latent: (B, latent_dim) state latent.
            action_idx: (B,) action index tensor.

        Returns:
            Tuple of (change_prob, next_latent_pred):
            - change_prob: (B, 1) sigmoid probability of frame change.
            - next_latent_pred: (B, latent_dim) predicted next latent.
        """
        act_emb = self.action_embed(action_idx)
        x = torch.cat([latent, act_emb], dim=-1)
        change_prob = torch.sigmoid(self.change_head(x))
        next_latent = self.forward_head(x)
        return change_prob, next_latent

    def add_transition(
        self,
        state_latent: np.ndarray,
        action: dict[str, Any],
        next_latent: np.ndarray,
        frame_changed: bool,
    ) -> None:
        """Add a transition to the replay buffer (with deduplication).

        Args:
            state_latent: Latent before the action.
            action: Action dict with "action" key and optional "data".
            next_latent: Latent after the action.
            frame_changed: Whether the frame visually changed.
        """
        # Deduplicate by hashing the state+action combination
        h = hashlib.md5(
            state_latent.tobytes() + action["action"].encode()
        ).hexdigest()

        if h in self._seen_hashes:
            return  # Skip duplicate
        self._seen_hashes.add(h)

        self._buffer.append(
            Transition(
                state_latent=state_latent.copy(),
                action=action["action"],
                action_data=action.get("data"),
                next_latent=next_latent.copy(),
                frame_changed=frame_changed,
            )
        )

    def train_step(self, batch_size: int = 64) -> float:
        """Perform one gradient step on a random batch from the buffer.

        Uses BCE loss for change prediction + MSE for forward model
        + entropy regularization on the change prediction.

        Args:
            batch_size: Mini-batch size.

        Returns:
            Total loss value.
        """
        if len(self._buffer) < batch_size:
            return 0.0

        # Sample random batch
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        latents = torch.from_numpy(np.stack([t.state_latent for t in batch])).to(self.device)
        next_latents = torch.from_numpy(np.stack([t.next_latent for t in batch])).to(self.device)
        action_indices = torch.tensor(
            [self.ACTION_TO_IDX.get(t.action, 0) for t in batch],
            dtype=torch.long,
            device=self.device,
        )
        changed = torch.tensor(
            [float(t.frame_changed) for t in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        self.train()
        self.optimizer.zero_grad()

        change_prob, next_latent_pred = self.forward(latents, action_indices)

        # BCE loss for change prediction
        bce_loss = nn.functional.binary_cross_entropy(change_prob, changed)

        # Entropy regularization (encourage calibrated probabilities)
        eps = 1e-7
        entropy = -(change_prob * torch.log(change_prob + eps) +
                    (1 - change_prob) * torch.log(1 - change_prob + eps))
        entropy_reg = -0.01 * entropy.mean()

        # MSE loss for forward model
        fwd_loss = nn.functional.mse_loss(next_latent_pred, next_latents)

        total_loss = bce_loss + entropy_reg + 0.5 * fwd_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self._train_steps += 1
        self._change_loss_avg = 0.99 * self._change_loss_avg + 0.01 * bce_loss.item()
        self._forward_loss_avg = 0.99 * self._forward_loss_avg + 0.01 * fwd_loss.item()

        self.eval()
        return total_loss.item()

    def predict_change(
        self, latent: np.ndarray, action: dict[str, Any]
    ) -> float:
        """Predict P(frame changes | state, action).

        Args:
            latent: State latent vector.
            action: Action dict.

        Returns:
            Probability of frame change (0–1).
        """
        with torch.no_grad():
            lat = torch.from_numpy(latent).unsqueeze(0).to(self.device)
            act_idx = torch.tensor(
                [self.ACTION_TO_IDX.get(action["action"], 0)],
                dtype=torch.long,
                device=self.device,
            )
            change_prob, _ = self.forward(lat, act_idx)
            return change_prob.item()

    def predict_next_latent(
        self, latent: np.ndarray, action: dict[str, Any]
    ) -> np.ndarray:
        """Predict the next latent given current state and action.

        Used by the planner as a cheap simulator.

        Args:
            latent: Current state latent.
            action: Action dict.

        Returns:
            Predicted next latent vector.
        """
        with torch.no_grad():
            lat = torch.from_numpy(latent).unsqueeze(0).to(self.device)
            act_idx = torch.tensor(
                [self.ACTION_TO_IDX.get(action["action"], 0)],
                dtype=torch.long,
                device=self.device,
            )
            _, next_latent = self.forward(lat, act_idx)
            return next_latent.squeeze(0).cpu().numpy()

    def prediction_error(
        self,
        state_latent: np.ndarray | None,
        action: dict[str, Any] | None,
        actual_next_latent: np.ndarray,
    ) -> float:
        """Compute prediction error (curiosity signal).

        Args:
            state_latent: Latent before action (None if first step).
            action: Action taken (None if first step).
            actual_next_latent: Actual next latent.

        Returns:
            L2 distance between predicted and actual next latent.
        """
        if state_latent is None or action is None:
            return 0.0
        predicted = self.predict_next_latent(state_latent, action)
        return float(np.linalg.norm(predicted - actual_next_latent))

    def confidence(self) -> float:
        """Estimate the world model's confidence.

        Returns a value in [0, 1] based on training progress and
        average change-prediction loss. Used by the agent to decide
        whether to use the planner or the reactive policy.

        Returns:
            Confidence score in [0, 1].
        """
        if self._train_steps < 10:
            return 0.0
        # Lower loss → higher confidence
        confidence = 1.0 / (1.0 + self._change_loss_avg * 5)
        return min(confidence, 1.0)

    @property
    def buffer_size_current(self) -> int:
        """Current number of transitions in the buffer."""
        return len(self._buffer)
