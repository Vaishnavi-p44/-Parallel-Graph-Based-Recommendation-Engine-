"""
Unit tests for UserItemGraph engine.
Tests graph construction, bipartite property, degree computations, and candidate extraction.
"""

import os
import pytest
import networkx as nx
from engine.graph_engine import UserItemGraph

@pytest.fixture
def sample_graph():
    graph = UserItemGraph(name="TestGraph")
    # Toy bipartite graph:
    # User 1 -> Item A (5.0), Item B (4.0)
    # User 2 -> Item A (4.0), Item B (5.0), Item C (3.0)
    # User 3 -> Item C (4.5), Item D (4.0)
    graph.add_interaction("U1", "ItemA", weight=5.0)
    graph.add_interaction("U1", "ItemB", weight=4.0)
    graph.add_interaction("U2", "ItemA", weight=4.0)
    graph.add_interaction("U2", "ItemB", weight=5.0)
    graph.add_interaction("U2", "ItemC", weight=3.0)
    graph.add_interaction("U3", "ItemC", weight=4.5)
    graph.add_interaction("U3", "ItemD", weight=4.0)

    graph.set_item_metadata("ItemA", {"title": "Alpha Movie", "genres": "Sci-Fi"})
    graph.set_item_metadata("ItemB", {"title": "Beta Movie", "genres": "Sci-Fi"})
    graph.set_item_metadata("ItemC", {"title": "Gamma Movie", "genres": "Action"})
    graph.set_item_metadata("ItemD", {"title": "Delta Movie", "genres": "Drama"})
    return graph

def test_graph_construction(sample_graph):
    assert len(sample_graph.all_users) == 3
    assert len(sample_graph.all_items) == 4
    assert sample_graph.get_user_items("U1") == {"ItemA", "ItemB"}
    assert sample_graph.get_item_users("ItemA") == {"U1", "U2"}
    assert sample_graph.get_edge_weight("U1", "ItemA") == 5.0

def test_is_bipartite(sample_graph):
    stats = sample_graph.get_graph_statistics()
    assert stats["is_bipartite"] is True
    assert stats["num_users"] == 3
    assert stats["num_items"] == 4
    assert stats["num_edges"] == 7

def test_candidate_items(sample_graph):
    # For U1, consumed = {ItemA, ItemB}.
    # Reachable unconsumed item via U2 is ItemC.
    candidates = sample_graph.get_candidate_items("U1", reachable_only=True)
    assert "ItemC" in candidates
    assert "ItemA" not in candidates
    assert "ItemB" not in candidates

def test_subgraph_extraction(sample_graph):
    sub = sample_graph.get_subgraph_data("U1", top_recommendations=["ItemC"])
    assert sub["target_user"] == "U1"
    node_ids = {n["id"] for n in sub["nodes"]}
    assert "U1" in node_ids
    assert "ItemA" in node_ids or "ItemB" in node_ids
    assert "ItemC" in node_ids

def test_cold_start_user(sample_graph):
    # Add isolated user with 0 interactions
    sample_graph.all_users.add("ColdUser")
    candidates = sample_graph.get_candidate_items("ColdUser", reachable_only=True)
    # Falls back to all unconsumed items
    assert len(candidates) == 4
