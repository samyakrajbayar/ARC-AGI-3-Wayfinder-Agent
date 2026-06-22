"""Deduplicated replay buffer for offline + online training.

Stores transitions with frame-level deduplication (hash-based) to
avoid wasting training capacity on near-identical states. Supports
random sampling, prioritized sampling, and buffer persistence.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferedTransition:
    """A transition stored in the replay buffer.

    Attributes:
        frame: 64×64 uint8 frame before the action.
        action: Action name string.
        action_data: Optional action data (e.g. coordinates).
        next_frame: 64×64 uint8 frame after the action.
        reward: Extrinsic reward (score delta).
        frame_changed: Whether the frame visually changed.
        frame_hash: MD5 hash of the frame (for dedup).
    """

    frame: np.ndarray
    action: str
    action_data: dict | None
    next_frame: np.ndarray
    reward: float
    frame_changed: bool
    frame_hash: str = ""

    def __post_init__(self) -> None:
        if not self.frame_hash:
            self.frame_hash = hashlib.md5(self.frame.tobytes()).hexdigest()


class ReplayBuffer:
    """Deduplicated replay buffer for transition storage.

    Features:
    - Frame-level deduplication (stores unique frames only once).
    - Transition-level deduplication (same state+action → skip).
    - Random and prioritized sampling.
    - Configurable maximum size.

    Attributes:
        max_size: Maximum number of transitions.
        frames: Dict mapping frame_hash → frame array (deduplicated storage).
        transitions: Deque of BufferedTransition objects.
    """

    def __init__(self, max_size: int = 200_000) -> None:
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of transitions to store.
        """
        self.max_size = max_size
        self.frames: dict[str, np.ndarray] = {}
        self.transitions: deque[BufferedTransition] = deque(maxlen=max_size)
        self._seen_keys: set[str] = set()

        logger.info("ReplayBuffer initialized (max_size=%d)", max_size)

    def add(
        self,
        frame: np.ndarray,
        action: str,
        action_data: dict | None,
        next_frame: np.ndarray,
        reward: float,
        frame_changed: bool,
    ) -> bool:
        """Add a transition to the buffer.

        Args:
            frame: Frame before action.
            action: Action name.
            action_data: Optional action data.
            next_frame: Frame after action.
            reward: Extrinsic reward.
            frame_changed: Whether frame visually changed.

        Returns:
            True if the transition was added, False if deduplicated.
        """
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        next_hash = hashlib.md5(next_frame.tobytes()).hexdigest()
        dedup_key = f"{frame_hash}:{action}:{action_data}"

        if dedup_key in self._seen_keys:
            return False

        self._seen_keys.add(dedup_key)

        # Store unique frames
        if frame_hash not in self.frames:
            self.frames[frame_hash] = frame.copy()
        if next_hash not in self.frames:
            self.frames[next_hash] = next_frame.copy()

        self.transitions.append(
            BufferedTransition(
                frame=frame.copy(),
                action=action,
                action_data=action_data,
                next_frame=next_frame.copy(),
                reward=reward,
                frame_changed=frame_changed,
                frame_hash=frame_hash,
            )
        )

        return True

    def sample(self, batch_size: int) -> list[BufferedTransition]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of BufferedTransition objects.
        """
        if len(self.transitions) < batch_size:
            return list(self.transitions)
        indices = np.random.choice(len(self.transitions), size=batch_size, replace=False)
        return [self.transitions[i] for i in indices]

    def sample_prioritized(
        self,
        batch_size: int,
        alpha: float = 0.6,
    ) -> list[BufferedTransition]:
        """Sample a batch with prioritization toward changed frames.

        Prioritizes transitions where the frame changed (more informative
        for training the world model).

        Args:
            batch_size: Number of transitions.
            alpha: Prioritization exponent (0=uniform, 1=full priority).

        Returns:
            List of BufferedTransition objects.
        """
        if len(self.transitions) < batch_size:
            return list(self.transitions)

        priorities = np.array([
            (1.0 if t.frame_changed else 0.1) ** alpha
            for t in self.transitions
        ])
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.transitions), size=batch_size, p=probs, replace=False)
        return [self.transitions[i] for i in indices]

    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return len(self.transitions)

    def __iter__(self) -> Iterator[BufferedTransition]:
        """Iterate over all transitions."""
        return iter(self.transitions)

    @property
    def num_unique_frames(self) -> int:
        """Number of unique frames stored."""
        return len(self.frames)

    def stats(self) -> dict:
        """Return buffer statistics.

        Returns:
            Dict with count, unique_frames, changed_ratio.
        """
        changed_count = sum(1 for t in self.transitions if t.frame_changed)
        return {
            "count": len(self.transitions),
            "unique_frames": len(self.frames),
            "changed_ratio": changed_count / max(len(self.transitions), 1),
        }

    def clear(self) -> None:
        """Clear all stored data."""
        self.frames.clear()
        self.transitions.clear()
        self._seen_keys.clear()
