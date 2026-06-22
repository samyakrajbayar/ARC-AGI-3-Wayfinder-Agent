"""Wayfinder agent for ARC-AGI-3.

A hybrid world-model + planning agent that combines:
- A learned perception encoder (CNN)
- A self-supervised world/transition model
- A hash-deduplicated state memory graph
- An intrinsic reward module (extrinsic + novelty + curiosity)
- A short-horizon tree-search planner
- A hierarchical action head (type selection + coordinate prediction)
"""

from agents.wayfinder.agent import WayfinderAgent

__all__ = ["WayfinderAgent"]
__version__ = "0.1.0"
