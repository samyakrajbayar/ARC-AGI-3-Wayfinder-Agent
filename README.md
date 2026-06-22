# ARC-AGI-3 Wayfinder Agent

Competition agent for the **ARC Prize 2026 — ARC-AGI-3 track** (Kaggle Code Competition).

## Architecture

The agent uses a **hybrid world-model + planning** architecture combining four
modules mapped to the benchmark's core capabilities:

| Module | Role | Capability |
|--------|------|------------|
| Perception Encoder | CNN over one-hot 64×64×16 frames → compact latent | — |
| World/Transition Model | Self-supervised P(frame changes | state, action) + forward model | Modeling |
| State Memory Graph | Hash-deduplicated directed graph of observed states | Exploration |
| Intrinsic Reward | Extrinsic Δscore + graph novelty + prediction-error curiosity | Goal-setting |
| Planner | Short-horizon tree search using world model as simulator | Planning |
| Action Head | Hierarchical: action-type softmax + conv coordinate head for ACTION6 | — |

## Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Run against a public game
uv run main.py --agent=wayfinder --game=ls20

# Run tests
uv run pytest

# Offline evaluation
uv run python eval/run_local_eval.py --agent=wayfinder --games=ls20,ls21,ls22
```

## Repository Layout

```
agents/wayfinder/     # Core agent modules (perception, world_model, memory_graph, ...)
training/             # Replay buffer, training loops, configs
eval/                 # Offline evaluation harness, metrics
notebooks/            # Kaggle submission notebook
tests/                # Unit + integration tests
```

## Key Constraints

- **No internet at inference time** — Kaggle scoring sessions disable network access.
- **MIT/CC0 license** — all authored code; third-party deps must be permissively licensed.
- **Action budget** — agent self-terminates stuck levels (~5× human median actions).
- **No per-game hardcoding** — same code runs against all unseen games.

## Reproducing Results

1. Install dependencies: `uv pip install -e ".[dev]"`
2. Download public games via the SDK's local mode.
3. Run evaluation: `uv run python eval/run_local_eval.py --agent=wayfinder`
4. Results are logged to `eval/results/` with per-game/level breakdowns.

## License

MIT — see [LICENSE](LICENSE).
