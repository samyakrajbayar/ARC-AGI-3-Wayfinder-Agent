"""Offline evaluation harness — runs the agent against local games.

Uses the toolkit's local-execution mode so iteration doesn't burn API
quota. Produces per-game/level score breakdowns and action-efficiency
metrics compared against random and human baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from agents.wayfinder.agent import WayfinderAgent
from eval.metrics import EvalMetrics, compute_efficiency

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("eval/results")


def run_local_eval(
    agent_name: str = "wayfinder",
    games: list[str] | None = None,
    max_actions_per_level: int = 1000,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> dict:
    """Run offline evaluation against local games.

    Args:
        agent_name: Name of the agent to evaluate.
        games: List of game IDs to evaluate. If None, uses defaults.
        max_actions_per_level: Max actions per level.
        device: Torch device.
        output_dir: Directory for output files.

    Returns:
        Evaluation results dict.
    """
    if games is None:
        games = ["ls20", "ls21", "ls22"]

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = WayfinderAgent(
        max_actions=max_actions_per_level,
        device=device,
    )

    all_results = {}

    for game_id in games:
        logger.info("Evaluating game: %s", game_id)
        game_result = _evaluate_game(agent, game_id, max_actions_per_level)
        all_results[game_id] = game_result

    # Compute aggregate metrics
    total_score = sum(r["score"] for r in all_results.values())
    total_actions = sum(r["total_actions"] for r in all_results.values())
    total_levels = sum(r["levels_attempted"] for r in all_results.values())
    levels_won = sum(r["levels_won"] for r in all_results.values())

    summary = {
        "agent": agent_name,
        "games": all_results,
        "aggregate": {
            "total_score": total_score,
            "total_actions": total_actions,
            "total_levels": total_levels,
            "levels_won": levels_won,
            "win_rate": levels_won / max(total_levels, 1),
            "avg_actions_per_level": total_actions / max(total_levels, 1),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save results
    results_path = output_dir / f"eval_{agent_name}_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return summary


def _evaluate_game(
    agent: WayfinderAgent,
    game_id: str,
    max_actions: int,
) -> dict:
    """Evaluate the agent on a single game.

    In production, this uses the SDK's local execution mode. For
    scaffolding/testing, it simulates with random frames.

    Args:
        agent: The agent to evaluate.
        game_id: Game identifier.
        max_actions: Max actions per level.

    Returns:
        Dict with score, actions, levels, etc.
    """
    agent.reset()

    metrics = EvalMetrics()
    levels_attempted = 0
    levels_won = 0
    total_score = 0.0
    total_actions = 0

    try:
        # Try to use the SDK's local execution
        from arc_agi_3 import LocalEnvironment  # type: ignore[import]

        env = LocalEnvironment(game_id=game_id)
        frames, state, score, win_score, available = env.reset()

        while state == "NOT_FINISHED":
            result = agent.act(
                frames=frames,
                state=state,
                score=score,
                win_score=win_score,
                available_actions=available,
            )
            frames, state, score, win_score, available = env.step(result)
            metrics.record_step(score, result["action"])
            total_actions += 1

            if agent.is_done(frames, state):
                break

        levels_attempted = 1
        levels_won = 1 if state == "WIN" else 0
        total_score = score

    except ImportError:
        # SDK not available — simulate
        logger.warning("SDK not available — running simulation for %s", game_id)

        for level in range(3):  # Simulate 3 levels
            agent.reset()
            levels_attempted += 1
            level_score = 0.0

            for step in range(max_actions):
                frame = np.random.randint(0, 16, size=(64, 64), dtype=np.uint8)
                state = "NOT_FINISHED"

                result = agent.act(
                    frames=[frame],
                    state=state,
                    score=level_score,
                    win_score=1.0,
                    available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
                )
                metrics.record_step(level_score, result["action"])
                total_actions += 1

                # Random chance of "winning" for simulation
                if np.random.random() < 0.01:
                    levels_won += 1
                    level_score = 1.0
                    break

                if agent.is_done([frame], state):
                    break

            total_score += level_score

    stats = metrics.compute()
    stats["efficiency"] = compute_efficiency(stats, max_actions)

    return {
        "score": total_score,
        "total_actions": total_actions,
        "levels_attempted": levels_attempted,
        "levels_won": levels_won,
        "metrics": stats,
    }


def main() -> int:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Run local evaluation")
    parser.add_argument("--agent", default="wayfinder")
    parser.add_argument("--games", default="ls20,ls21,ls22", help="Comma-separated game IDs")
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = run_local_eval(
        agent_name=args.agent,
        games=args.games.split(","),
        max_actions_per_level=args.max_actions,
        device=args.device,
        output_dir=Path(args.output_dir),
    )

    # Print summary
    agg = results["aggregate"]
    print("\n" + "=" * 60)
    print(f"Agent: {results['agent']}")
    print(f"Games: {len(results['games'])}")
    print(f"Levels: {agg['total_levels']} (won: {agg['levels_won']})")
    print(f"Win rate: {agg['win_rate']:.1%}")
    print(f"Total score: {agg['total_score']:.2f}")
    print(f"Total actions: {agg['total_actions']}")
    print(f"Avg actions/level: {agg['avg_actions_per_level']:.1f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
