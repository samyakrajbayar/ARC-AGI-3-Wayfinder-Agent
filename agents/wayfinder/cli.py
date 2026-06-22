"""CLI entry point for the Wayfinder agent.

Wraps the SDK's main.py interface so the agent can be run via:
    uv run main.py --agent=wayfinder --game=ls20
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Wayfinder agent for ARC-AGI-3"
    )
    parser.add_argument(
        "--agent", default="wayfinder",
        help="Agent name (default: wayfinder)"
    )
    parser.add_argument(
        "--game", default="ls20",
        help="Game ID to play (default: ls20)"
    )
    parser.add_argument(
        "--max-actions", type=int, default=1000,
        help="Max actions per level (default: 1000)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (default: cpu)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Try to use the SDK's runner
    try:
        from arc_agi_3 import main as sdk_main  # type: ignore[import]
        logger.info("Using SDK runner with agent=%s, game=%s", args.agent, args.game)
        # The SDK's main.py handles the game loop
        sdk_main(agent_name=args.agent, game_id=args.game)
    except ImportError:
        logger.error(
            "arc-agi-3 SDK not found. Install with: uv pip install arc-agi-3"
        )
        logger.info("Running in standalone mode (no SDK)...")
        _run_standalone(args)

    return 0


def _run_standalone(args: argparse.Namespace) -> None:
    """Run the agent without the SDK (for testing/scaffolding).

    Args:
        args: Parsed CLI arguments.
    """
    import numpy as np

    from agents.wayfinder.agent import WayfinderAgent

    agent = WayfinderAgent(
        max_actions=args.max_actions,
        device=args.device,
    )

    # Simulate a few steps with random frames
    logger.info("Running standalone simulation (5 steps)...")
    for step in range(5):
        frame = np.random.randint(0, 16, size=(64, 64), dtype=np.uint8)
        result = agent.act(
            frames=[frame],
            state="NOT_FINISHED",
            score=0.0,
            win_score=1.0,
            available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
        )
        logger.info("Step %d: %s", step + 1, result)

    logger.info("Standalone simulation complete.")


if __name__ == "__main__":
    sys.exit(main())
