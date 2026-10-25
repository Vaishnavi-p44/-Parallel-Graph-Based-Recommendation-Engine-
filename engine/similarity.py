"""
Graph-based Recommendation Similarity Metrics:
1. Common Neighbors (Bipartite 3-hop Path Scoring / Co-occurrence)
2. Jaccard Similarity (Normalized Neighborhood Overlap)

Provides pure, high-performance mathematical functions that operate directly
on dictionary-of-sets adjacency structures.
"""

from typing import Dict, Set, Tuple, List, Optional, Any

def common_neighbors_user_similarity(
    user_u_items: Set[str],
    user_v_items: Set[str]
) -> int:
    """
    Computes Common Neighbors between two users: |N(u) ∩ N(v)|
    Represents the count of shared items both users have interacted with.
    """
    return len(user_u_items.intersection(user_v_items))

def jaccard_user_similarity(
    user_u_items: Set[str],
    user_v_items: Set[str]
) -> float:
    """
    Computes Jaccard Similarity between two users:
    J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    Normalized between [0.0, 1.0].
    """
    if not user_u_items or not user_v_items:
        return 0.0

    intersection_len = len(user_u_items.intersection(user_v_items))
    if intersection_len == 0:
        return 0.0

    union_len = len(user_u_items.union(user_v_items))
    return float(intersection_len) / float(union_len)

def score_candidate_common_neighbors(
    target_user: str,
    candidate_item: str,
    user_u_items: Set[str],
    item_users: Set[str],
    user_neighbors: Dict[str, Set[str]],
    edge_weights: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes candidate item recommendation score using Common Neighbors:
    Score(u, item) = Σ_{v ∈ N(item), v ≠ u} [ |N(u) ∩ N(v)| × weight(v, item) ]

    Returns:
        (total_score, details_dict)
    """
    total_score = 0.0
    contributing_users = []

    for other_user in item_users:
        if other_user == target_user:
            continue

        v_items = user_neighbors.get(other_user, set())
        shared = user_u_items.intersection(v_items)
        cn = len(shared)

        if cn > 0:
            weight = 1.0
            if edge_weights:
                weight = edge_weights.get((other_user, candidate_item), 1.0)

            score_contribution = cn * weight
            total_score += score_contribution

            contributing_users.append({
                "user_id": other_user,
                "similarity": cn,
                "shared_items": list(shared)[:4],
                "shared_count": cn,
                "rating": weight
            })

    # Sort contributors by highest similarity
    contributing_users.sort(key=lambda x: x["similarity"], reverse=True)

    details = {
        "algorithm": "common_neighbors",
        "contributing_users_count": len(contributing_users),
        "top_contributors": contributing_users[:3],
        "raw_score": round(total_score, 4)
    }

    return total_score, details

def score_candidate_jaccard(
    target_user: str,
    candidate_item: str,
    user_u_items: Set[str],
    item_users: Set[str],
    user_neighbors: Dict[str, Set[str]],
    edge_weights: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes candidate item recommendation score using Jaccard Similarity:
    Score(u, item) = Σ_{v ∈ N(item), v ≠ u} [ Jaccard(u, v) × weight(v, item) ]

    Returns:
        (total_score, details_dict)
    """
    total_score = 0.0
    contributing_users = []

    if not user_u_items:
        return 0.0, {"algorithm": "jaccard", "contributing_users_count": 0, "top_contributors": [], "raw_score": 0.0}

    for other_user in item_users:
        if other_user == target_user:
            continue

        v_items = user_neighbors.get(other_user, set())
        if not v_items:
            continue

        shared = user_u_items.intersection(v_items)
        if not shared:
            continue

        intersection_len = len(shared)
        union_len = len(user_u_items.union(v_items))
        jaccard_val = float(intersection_len) / float(union_len)

        weight = 1.0
        if edge_weights:
            weight = edge_weights.get((other_user, candidate_item), 1.0)

        score_contribution = jaccard_val * weight
        total_score += score_contribution

        contributing_users.append({
            "user_id": other_user,
            "similarity": round(jaccard_val, 4),
            "shared_items": list(shared)[:4],
            "shared_count": intersection_len,
            "rating": weight
        })

    contributing_users.sort(key=lambda x: x["similarity"], reverse=True)

    details = {
        "algorithm": "jaccard",
        "contributing_users_count": len(contributing_users),
        "top_contributors": contributing_users[:3],
        "raw_score": round(total_score, 4)
    }

    return total_score, details
