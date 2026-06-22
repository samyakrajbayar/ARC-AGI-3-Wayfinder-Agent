"""Main agent class — glue connecting all modules.

Subclasses the official ARC-AGI-3 SDK Agent base class, wiring together
perception, world model, memory graph, intrinsic reward, planner, and
action head into a single play loop.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agents.wayfinder.action_head import ActionHead
from agents.wayfinder.intrinsic_reward import IntrinsicReward
from agents.wayfinder.memory_graph import MemoryGraph
from agents.wayfinder.perception import PerceptionEncoder
from agents.wayfinder.planner import Planner
from agents.wayfinder.world_model import WorldModel

logger = logging.getLogger(__name__)


class WayfinderAgent:
    """Hybrid world-model + planning agent for ARC-AGI-3.

    The agent maintains:
    - A perception encoder that converts raw 64×64×4-bit frames into latents.
    - A world model that predicts frame changes and forward dynamics.
    - A memory graph that tracks visited states for novelty.
    - An intrinsic reward module combining extrinsic score, novelty, curiosity.
    - A planner that does short-horizon tree search using the world model.
    - An action head with hierarchical action-type + coordinate selection.

    The main loop per step:
    1. Encode current frame → latent + diff mask
    2. Update memory graph with the new state
    3. Train world model on the latest transition (online)
    4. Compute intrinsic reward for the current state
    5. If world model confidence is high, use planner; else use reactive policy
    6. Select action via action head (hierarchical)
    7. Return the action and reasoning blob for audit

    Attributes:
        max_actions: Maximum actions before self-terminating a level.
        action_count: Actions taken in the current level attempt.
        _encoder: Perception encoder module.
        _world_model: World/transition model.
        _memory: State memory graph.
        _reward: Intrinsic reward calculator.
        _planner: Short-horizon tree search planner.
        _action_head: Hierarchical action selection head.
    """

    def __init__(
        self,
        max_actions: int = 1000,
        latent_dim: int = 256,
        buffer_size: int = 200_000,
        device: str = "cpu",
    ) -> None:
        """Initialize the Wayfinder agent.

        Args:
            max_actions: Safety cap on actions per level attempt.
            latent_dim: Size of the perception encoder's output latent.
            buffer_size: Maximum transitions stored in the replay buffer.
            device: Torch device ("cpu" or "cuda").
        """
        self.max_actions = max_actions
        self.action_count = 0
        self.current_score: float = 0.0
        self.win_threshold: float = 1.0
        self._device = device

        # Module A: Perception encoder
        self._encoder = PerceptionEncoder(latent_dim=latent_dim, device=device)

        # Module B: World/transition model
        self._world_model = WorldModel(
            latent_dim=latent_dim,
            buffer_size=buffer_size,
            device=device,
        )

        # Module C: State memory graph
        self._memory = MemoryGraph()

        # Module D: Intrinsic reward
        self._reward = IntrinsicReward()

        # Module E: Planner
        self._planner = Planner(
            world_model=self._world_model,
            reward_module=self._reward,
            max_depth=5,
            max_simulations=50,
        )

        # Module F: Action head
        self._action_head = ActionHead(
            latent_dim=latent_dim,
            device=device,
        )

        # Previous state tracking
        self._prev_latent: np.ndarray | None = None
        self._prev_frame_hash: str | None = None
        self._is_done = False

        logger.info("WayfinderAgent initialized (device=%s, max_actions=%d)", device, max_actions)

    def is_done(self, frames: list[np.ndarray], state: str, **kwargs: Any) -> bool:
        """Check if the agent should stop playing the current level.

        The agent self-terminates when:
        - The game state is WIN or GAME_OVER
        - The action budget is exhausted
        - The session is NOT_STARTED (needs RESET)

        Args:
            frames: List of 64×64 frames from the latest step.
            state: Current game state string.
            **kwargs: Additional context (score, etc.).

        Returns:
            True if the agent should stop, False to continue.
        """
        if state in ("WIN", "GAME_OVER", "NOT_STARTED"):
            self._is_done = True
            return True

        if self.action_count >= self.max_actions:
            logger.warning(
                "Action budget exhausted (%d/%d) — self-terminating level",
                self.action_count,
                self.max_actions,
            )
            self._is_done = True
            return True

        return False

    def act(
        self,
        frames: list[np.ndarray],
        state: str,
        score: float,
        win_score: float,
        available_actions: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Select the next action given the current observation.

        This is the main per-step entry point called by the SDK harness.

        Args:
            frames: List of 64×64 numpy arrays (4-bit color indices 0–15).
            state: Game state string ("NOT_FINISHED", etc.).
            score: Current running score.
            win_score: Score threshold to win the level.
            available_actions: List of action names available this step.
            **kwargs: Additional SDK fields (levels_completed, etc.).

        Returns:
            Dict with "action" (str), "data" (dict, for ACTION6 x/y),
            and "reasoning" (dict, for audit/logging).
        """
        self.action_count += 1
        self.current_score = score
        self.win_threshold = win_score

        # Use the last frame if multiple are returned (animation settled)
        current_frame = frames[-1] if frames else np.zeros((64, 64), dtype=np.uint8)

        # --- Step 1: Encode frame ---
        latent, diff_mask = self._encoder.encode(current_frame)
        frame_hash = self._memory.hash_frame(current_frame)

        # --- Step 2: Update memory graph ---
        score_delta = score - self.current_score if self._prev_latent is not None else 0.0
        if self._prev_frame_hash is not None:
            # We don't know the action yet — we'll record the edge after
            # the action is chosen. For now, register the node.
            pass
        self._memory.add_node(frame_hash, latent, score)

        novelty = self._memory.novelty(frame_hash)

        # --- Step 3: Online world-model update ---
        if self._prev_latent is not None and self._last_action is not None:
            self._world_model.add_transition(
                state_latent=self._prev_latent,
                action=self._last_action,
                next_latent=latent,
                frame_changed=bool(np.any(diff_mask)),
            )
            self._world_model.train_step()

        # --- Step 4: Intrinsic reward ---
        prediction_error = self._world_model.prediction_error(
            self._prev_latent, self._last_action, latent
        ) if self._prev_latent is not None and self._last_action else 0.0

        utility = self._reward.compute(
            extrinsic_delta=score_delta,
            novelty=novelty,
            prediction_error=prediction_error,
        )

        # --- Step 5: Plan or react ---
        model_confidence = self._world_model.confidence()
        if model_confidence > 0.65 and self.action_count > 10:
            # Use planner when the world model is confident
            action = self._planner.plan(
                latent=latent,
                available_actions=available_actions,
                action_budget_remaining=self.max_actions - self.action_count,
                utility_fn=self._reward.compute,
            )
        else:
            # Reactive policy via action head
            action = self._action_head.select(
                latent=latent,
                diff_mask=diff_mask,
                available_actions=available_actions,
                world_model=self._world_model,
                epsilon=0.15 if novelty > 0.5 else 0.05,
            )

        # --- Step 6: Record transition in memory graph ---
        if self._prev_frame_hash is not None:
            self._memory.add_edge(
                from_hash=self._prev_frame_hash,
                to_hash=frame_hash,
                action=action["action"],
                action_data=action.get("data", {}),
            )

        # --- Step 7: Update previous state ---
        self._prev_latent = latent
        self._prev_frame_hash = frame_hash
        self._last_action = action

        # Build reasoning blob for audit (≤16 KB)
        reasoning = {
            "step": self.action_count,
            "score": score,
            "win_score": win_score,
            "model_confidence": model_confidence,
            "novelty": novelty,
            "prediction_error": prediction_error,
            "utility": utility,
            "mode": "plan" if model_confidence > 0.65 else "react",
            "frame_hash": frame_hash[:16],
        }

        logger.debug("Step %d: action=%s, reasoning=%s", self.action_count, action["action"], reasoning)

        return {"action": action["action"], "data": action.get("data", {}), "reasoning": reasoning}

    def reset(self) -> None:
        """Reset agent state for a new level attempt.

        Clears per-level state: action count, memory graph, previous latents.
        The world model and perception encoder retain their learned weights
        across resets (they generalize across levels within a game).
        """
        self.action_count = 0
        self.current_score = 0.0
        self._prev_latent = None
        self._prev_frame_hash = None
        self._last_action: dict[str, Any] | None = None
        self._is_done = False
        self._memory.reset()
        logger.info("Agent reset for new level attempt")

    # Expose _last_action with a default for the first step
    _last_action: dict[str, Any] | None = None
