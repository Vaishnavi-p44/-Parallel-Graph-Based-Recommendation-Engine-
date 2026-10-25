"""
Unit tests for Common Neighbors and Jaccard Similarity mathematical formulations.
"""

import pytest
from engine.similarity import (
    common_neighbors_user_similarity,
    jaccard_user_similarity,
    score_candidate_common_neighbors,
    score_candidate_jaccard
)

def test_common_neighbors_math():
    set_a = {"item1", "item2", "item3"}
    set_b = {"item2", "item3", "item4"}
    set_c = {"item5", "item6"}

    # Overlap between A and B is {item2, item3} -> 2
    assert common_neighbors_user_similarity(set_a, set_b) == 2
    # Overlap between A and C is empty -> 0
    assert common_neighbors_user_similarity(set_a, set_c) == 0

def test_jaccard_similarity_math():
    set_a = {"item1", "item2", "item3"}
    set_b = {"item2", "item3", "item4"}
    set_c = {"item5", "item6"}

    # Intersection: 2, Union: 4 -> Jaccard = 2/4 = 0.5
    assert jaccard_user_similarity(set_a, set_b) == 0.5
    # Disjoint sets -> Jaccard = 0.0
    assert jaccard_user_similarity(set_a, set_c) == 0.0
    # Identical sets -> Jaccard = 1.0
    assert jaccard_user_similarity(set_a, set_a) == 1.0
    # Empty sets
    assert jaccard_user_similarity(set(), set_a) == 0.0

def test_score_candidate_common_neighbors():
    user_u_items = {"item1", "item2"}
    candidate_item = "item3"
    item_users = {"user2", "user3"}

    user_neighbors = {
        "user1": {"item1", "item2"},
        "user2": {"item1", "item2", "item3"}, # Shares 2 items with user1
        "user3": {"item2", "item3"}            # Shares 1 item with user1
    }

    edge_weights = {
        ("user2", "item3"): 1.0,
        ("user3", "item3"): 2.0
    }

    score, details = score_candidate_common_neighbors(
        target_user="user1",
        candidate_item=candidate_item,
        user_u_items=user_u_items,
        item_users=item_users,
        user_neighbors=user_neighbors,
        edge_weights=edge_weights
    )

    # Score = (2 * 1.0) + (1 * 2.0) = 4.0
    assert score == 4.0
    assert details["contributing_users_count"] == 2
    assert details["top_contributors"][0]["user_id"] == "user2"

def test_score_candidate_jaccard():
    user_u_items = {"item1", "item2"}
    candidate_item = "item3"
    item_users = {"user2"}

    # user1 items: {1, 2}
    # user2 items: {1, 2, 3}
    # Intersection = 2, Union = 3 -> Jaccard = 2/3 = 0.6667
    user_neighbors = {
        "user1": {"item1", "item2"},
        "user2": {"item1", "item2", "item3"}
    }
    edge_weights = {("user2", "item3"): 1.0}

    score, details = score_candidate_jaccard(
        target_user="user1",
        candidate_item=candidate_item,
        user_u_items=user_u_items,
        item_users=item_users,
        user_neighbors=user_neighbors,
        edge_weights=edge_weights
    )

    assert round(score, 3) == 0.667
    assert details["contributing_users_count"] == 1
