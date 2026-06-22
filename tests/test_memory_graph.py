"""Tests for the memory graph module."""

from __future__ import annotations

import numpy as np
import pytest

from agents.wayfinder.memory_graph import MemoryGraph
from agents.wayfinder.perception import GRID_SIZE


class TestMemoryGraph:
    """Test cases for MemoryGraph."""

    @pytest.fixture
    def graph(self) -> MemoryGraph:
        """Create a test memory graph."""
        return MemoryGraph()

    @pytest.fixture
    def sample_frame(self) -> np.ndarray:
        """Create a sample frame."""
        return np.random.randint(0, 16, size=(GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    def test_hash_frame_stable(self, graph: MemoryGraph, sample_frame: np.ndarray) -> None:
        """Test that hashing is deterministic."""
        h1 = graph.hash_frame(sample_frame)
        h2 = graph.hash_frame(sample_frame)
        assert h1 == h2

    def test_hash_frame_different_for_different_frames(self, graph: MemoryGraph) -> None:
        """Test that different frames have different hashes."""
        f1 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        f2 = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        assert graph.hash_frame(f1) != graph.hash_frame(f2)

    def test_add_node_new(self, graph: MemoryGraph, sample_frame: np.ndarray) -> None:
        """Test adding a new node."""
        h = graph.hash_frame(sample_frame)
        latent = np.random.randn(64).astype(np.float32)
        graph.add_node(h, latent, score=0.5)
        assert h in graph.nodes
        assert graph.nodes[h]["visit_count"] == 1
        assert graph.nodes[h]["score"] == 0.5

    def test_add_node_existing_increments_visits(self, graph: MemoryGraph, sample_frame: np.ndarray) -> None:
        """Test that re-adding a node increments its visit count."""
        h = graph.hash_frame(sample_frame)
        latent = np.random.randn(64).astype(np.float32)
        graph.add_node(h, latent)
        graph.add_node(h, latent)
        graph.add_node(h, latent)
        assert graph.nodes[h]["visit_count"] == 3

    def test_novelty_new_node(self, graph: MemoryGraph) -> None:
        """Test novelty is 1.0 for unseen nodes."""
        assert graph.novelty("nonexistent") == 1.0

    def test_novelty_decreases_with_visits(self, graph: MemoryGraph, sample_frame: np.ndarray) -> None:
        """Test that novelty decreases as visit count increases."""
        h = graph.hash_frame(sample_frame)
        latent = np.random.randn(64).astype(np.float32)
        graph.add_node(h, latent)
        n1 = graph.novelty(h)
        graph.add_node(h, latent)
        n2 = graph.novelty(h)
        assert n2 < n1
        assert 0 < n2 < 1

    def test_add_edge(self, graph: MemoryGraph, sample_frame: np.ndarray) -> None:
        """Test adding an edge between nodes."""
        h1 = graph.hash_frame(sample_frame)
        h2 = graph.hash_frame(sample_frame + 1)
        latent = np.random.randn(64).astype(np.float32)
        graph.add_node(h1, latent)
        graph.add_node(h2, latent)
        graph.add_edge(h1, h2, "ACTION1")

        assert (h1, h2) in graph.edges
        assert graph.edges[(h1, h2)]["action"] == "ACTION1"
        assert len(graph.adjacency[h1]) == 1

    def test_reset_clears_graph(self, graph: MemoryGraph, sample_frame: np.ndarray) -> None:
        """Test that reset clears all nodes and edges."""
        h = graph.hash_frame(sample_frame)
        latent = np.random.randn(64).astype(np.float32)
        graph.add_node(h, latent)
        graph.add_edge(h, h, "ACTION1")

        graph.reset()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_stats(self, graph: MemoryGraph, sample_frame: np.ndarray) -> None:
        """Test stats returns correct summary."""
        h = graph.hash_frame(sample_frame)
        latent = np.random.randn(64).astype(np.float32)
        graph.add_node(h, latent)
        graph.add_node(h, latent)
        stats = graph.stats()
        assert stats["node_count"] == 1
        assert stats["avg_visits"] == 2.0
