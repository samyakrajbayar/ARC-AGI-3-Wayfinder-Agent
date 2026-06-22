"""State memory graph — Module C.

A hash-deduplicated directed graph of observed game states. Nodes are
hashed frames; edges are actions taken. Tracks visit counts for novelty
detection and supports backtracking/loop avoidance.

Inspired by Blind Squirrel's graph-based exploration, but integrated
with the learned perception encoder for efficient similarity checks.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class MemoryGraph:
    """Directed graph of observed game states.

    Nodes are identified by frame hashes (MD5 of the raw 64×64 frame).
    Each node stores:
    - A latent vector (from the perception encoder) for similarity checks.
    - The score observed at that state.
    - A visit count (incremented each time the state is seen).

    Edges store the action that caused the transition.

    The graph resets per RESET (per level attempt), but the agent can
    optionally carry summary statistics across attempts.

    Attributes:
        nodes: Dict mapping frame_hash → node data.
        edges: Dict mapping (from_hash, to_hash) → edge data.
        adjacency: Dict mapping from_hash → list of (action, to_hash).
    """

    def __init__(self) -> None:
        """Initialize an empty memory graph."""
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str], dict] = {}
        self.adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)

    @staticmethod
    def hash_frame(frame: np.ndarray) -> str:
        """Compute a stable hash for a 64×64 frame.

        Args:
            frame: 64×64 uint8 array of color indices.

        Returns:
            MD5 hex string of the frame data.
        """
        return hashlib.md5(frame.tobytes()).hexdigest()

    def add_node(
        self,
        frame_hash: str,
        latent: np.ndarray,
        score: float = 0.0,
    ) -> None:
        """Add or update a node in the graph.

        If the node already exists, increment its visit count.

        Args:
            frame_hash: Hash of the frame.
            latent: Latent vector from the perception encoder.
            score: Game score at this state.
        """
        if frame_hash in self.nodes:
            self.nodes[frame_hash]["visit_count"] += 1
            self.nodes[frame_hash]["score"] = score
        else:
            self.nodes[frame_hash] = {
                "latent": latent.copy(),
                "score": score,
                "visit_count": 1,
            }

    def add_edge(
        self,
        from_hash: str,
        to_hash: str,
        action: str,
        action_data: dict | None = None,
    ) -> None:
        """Add a directed edge to the graph.

        Args:
            from_hash: Source node hash.
            to_hash: Destination node hash.
            action: Action name that caused this transition.
            action_data: Optional action data (e.g. coordinates for ACTION6).
        """
        edge_key = (from_hash, to_hash)
        if edge_key in self.edges:
            self.edges[edge_key]["count"] += 1
        else:
            self.edges[edge_key] = {
                "action": action,
                "action_data": action_data,
                "count": 1,
            }
            self.adjacency[from_hash].append((action, to_hash))

    def novelty(self, frame_hash: str) -> float:
        """Compute novelty score for a frame.

        Novelty is high for unvisited or rarely-visited states, low for
        frequently-visited ones. Used by the intrinsic reward module.

        Formula: novelty = 1.0 / (1.0 + visit_count)

        Args:
            frame_hash: Hash of the frame to evaluate.

        Returns:
            Novelty score in (0, 1]. New states return 1.0.
        """
        if frame_hash not in self.nodes:
            return 1.0
        visit_count = self.nodes[frame_hash]["visit_count"]
        return 1.0 / (1.0 + visit_count)

    def shortest_path_to_unexplored(self, from_hash: str) -> list[str] | None:
        """Find shortest path from a node to any unexplored action.

        Uses BFS to find the nearest node that has untried actions
        (actions not yet taken from that node).

        Args:
            from_hash: Starting node hash.

        Returns:
            List of action names forming the path, or None if no
            unexplored action is reachable.
        """
        if from_hash not in self.nodes:
            return None

        # BFS
        from collections import deque

        queue: deque[tuple[str, list[str]]] = deque([(from_hash, [])])
        visited: set[str] = {from_hash}

        all_actions = {"RESET", "ACTION1", "ACTION2", "ACTION3",
                       "ACTION4", "ACTION5", "ACTION6"}

        while queue:
            current_hash, path = queue.popleft()

            # Check if current node has unexplored actions
            tried_actions = {
                edge_action
                for edge_action, _ in self.adjacency.get(current_hash, [])
            }
            unexplored = all_actions - tried_actions

            if unexplored and len(path) > 0:
                return path
            if unexplored:
                # We're at the start node with unexplored actions
                return []

            # Expand neighbors
            for edge_action, neighbor_hash in self.adjacency.get(current_hash, []):
                if neighbor_hash not in visited:
                    visited.add(neighbor_hash)
                    queue.append((neighbor_hash, path + [edge_action]))

        return None

    def get_visited_states(self) -> set[str]:
        """Return the set of all visited node hashes.

        Returns:
            Set of frame hashes that have been visited.
        """
        return set(self.nodes.keys())

    def get_transition_count(self) -> int:
        """Return total number of unique transitions (edges).

        Returns:
            Number of edges in the graph.
        """
        return len(self.edges)

    def stats(self) -> dict:
        """Return summary statistics about the graph.

        Returns:
            Dict with node_count, edge_count, avg_visits, max_visits.
        """
        visit_counts = [n["visit_count"] for n in self.nodes.values()]
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "avg_visits": np.mean(visit_counts) if visit_counts else 0.0,
            "max_visits": max(visit_counts) if visit_counts else 0,
        }

    def reset(self) -> None:
        """Clear all nodes and edges.

        Called when the agent RESETs a level or starts a new one.
        """
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        logger.debug("Memory graph reset")

    def serialize(self) -> dict:
        """Serialize the graph for checkpointing.

        Returns:
            Dict representation of the graph.
        """
        return {
            "nodes": {
                h: {"score": n["score"], "visit_count": n["visit_count"]}
                for h, n in self.nodes.items()
            },
            "edges": {
                f"{k[0]}->{k[1]}": v for k, v in self.edges.items()
            },
        }
