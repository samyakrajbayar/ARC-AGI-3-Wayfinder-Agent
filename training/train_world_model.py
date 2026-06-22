"""Offline + online training loop for the world model.

This script can run in two modes:
1. Offline: Train on pre-collected transitions from a replay buffer file.
2. Online: Play games and train the world model incrementally.

Usage:
    uv run python training/train_world_model.py --mode offline --buffer data/buffer.pkl
    uv run python training/train_world_model.py --mode online --games ls20,ls21
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from agents.wayfinder.perception import PerceptionEncoder
from agents.wayfinder.world_model import WorldModel
from training.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def train_offline(
    buffer_path: str,
    latent_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cpu",
    save_path: str = "models/world_model.pt",
) -> None:
    """Train the world model offline on a pre-collected buffer.

    Args:
        buffer_path: Path to a pickled ReplayBuffer.
        latent_dim: Latent dimension.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        device: Torch device.
        save_path: Where to save the trained model.
    """
    logger.info("Loading replay buffer from %s", buffer_path)
    with open(buffer_path, "rb") as f:
        buffer: ReplayBuffer = pickle.load(f)

    logger.info("Buffer loaded: %d transitions, %d unique frames", len(buffer), buffer.num_unique_frames)

    encoder = PerceptionEncoder(latent_dim=latent_dim, device=device)
    world_model = WorldModel(latent_dim=latent_dim, device=device, lr=lr)

    # Pre-encode all unique frames
    logger.info("Encoding unique frames...")
    frame_hashes = list(buffer.frames.keys())
    frame_arrays = np.stack([buffer.frames[h] for h in frame_hashes])
    latents = encoder.encode_batch(frame_arrays)
    hash_to_latent = {h: lat for h, lat in zip(frame_hashes, latents)}

    logger.info("Training for %d epochs...", epochs)
    for epoch in range(epochs):
        batch = buffer.sample_prioritized(batch_size)
        total_loss = 0.0

        for t in batch:
            state_latent = hash_to_latent.get(t.frame_hash)
            next_hash = hash_to_latent.get(
                __import__("hashlib").md5(t.next_frame.tobytes()).hexdigest()
            )
            if state_latent is None or next_hash is None:
                continue

            world_model.add_transition(
                state_latent=state_latent,
                action={"action": t.action, "data": t.action_data},
                next_latent=next_hash,
                frame_changed=t.frame_changed,
            )
            loss = world_model.train_step(batch_size=min(batch_size, len(world_model._buffer)))
            total_loss += loss

        avg_loss = total_loss / max(len(batch), 1)
        if (epoch + 1) % 5 == 0:
            logger.info(
                "Epoch %d/%d: avg_loss=%.4f, buffer=%d, confidence=%.3f",
                epoch + 1, epochs, avg_loss,
                world_model.buffer_size_current,
                world_model.confidence(),
            )

    # Save model
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({
        "world_model": world_model.state_dict(),
        "encoder": encoder.state_dict(),
        "latent_dim": latent_dim,
    }, save_path)
    logger.info("Model saved to %s", save_path)


def train_online(
    games: list[str],
    max_actions_per_game: int = 500,
    latent_dim: int = 256,
    device: str = "cpu",
    save_path: str = "models/world_model_online.pt",
) -> None:
    """Train the world model online by playing games.

    Args:
        games: List of game IDs to play.
        max_actions_per_game: Max actions per game.
        latent_dim: Latent dimension.
        device: Torch device.
        save_path: Where to save the model.
    """
    from agents.wayfinder.agent import WayfinderAgent

    agent = WayfinderAgent(
        max_actions=max_actions_per_game,
        latent_dim=latent_dim,
        device=device,
    )

    for game_id in games:
        logger.info("Playing game %s...", game_id)
        agent.reset()

        # In real usage, this would use the SDK to play the game.
        # For now, we simulate with random frames.
        for step in range(max_actions_per_game):
            frame = np.random.randint(0, 16, size=(64, 64), dtype=np.uint8)
            result = agent.act(
                frames=[frame],
                state="NOT_FINISHED",
                score=0.0,
                win_score=1.0,
                available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
            )

            if agent.is_done([frame], "NOT_FINISHED"):
                break

        logger.info(
            "Game %s: %d actions, buffer=%d, confidence=%.3f",
            game_id, agent.action_count,
            agent._world_model.buffer_size_current,
            agent._world_model.confidence(),
        )

    import torch
    torch.save({
        "world_model": agent._world_model.state_dict(),
        "encoder": agent._encoder.state_dict(),
        "latent_dim": latent_dim,
    }, save_path)
    logger.info("Online model saved to %s", save_path)


def main() -> int:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train the world model")
    parser.add_argument("--mode", choices=["offline", "online"], default="online")
    parser.add_argument("--buffer", default="data/buffer.pkl", help="Path to replay buffer (offline mode)")
    parser.add_argument("--games", default="ls20,ls21,ls22", help="Comma-separated game IDs (online mode)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default="models/world_model.pt")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "offline":
        train_offline(
            buffer_path=args.buffer,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_path=args.save_path,
        )
    else:
        train_online(
            games=args.games.split(","),
            max_actions_per_game=500,
            latent_dim=args.latent_dim,
            device=args.device,
            save_path=args.save_path,
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
