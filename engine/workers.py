"""
Picklable Worker Tasks for Multiprocessing Master-Worker Architecture.
Specifically designed for Windows 'spawn' compatibility and minimal IPC overhead.
Worker processes initialize with local references to graph adjacency tables.
"""

from typing import Dict, Set, Tuple, List, Any
from engine.similarity import score_candidate_common_neighbors, score_candidate_jaccard

# Process-local cache populated by worker initializer
_LOCAL_USER_NEIGHBORS: Dict[str, Set[str]] = {}
_LOCAL_ITEM_NEIGHBORS: Dict[str, Set[str]] = {}
_LOCAL_EDGE_WEIGHTS: Dict[Tuple[str, str], float] = {}

def init_worker(
    user_neighbors: Dict[str, Set[str]],
    item_neighbors: Dict[str, Set[str]],
    edge_weights: Dict[Tuple[str, str], float]
):
    """
    Worker initializer executed once per process when a Pool starts.
    Stores read-only adjacency structures in worker process memory.
    """
    global _LOCAL_USER_NEIGHBORS, _LOCAL_ITEM_NEIGHBORS, _LOCAL_EDGE_WEIGHTS
    _LOCAL_USER_NEIGHBORS = user_neighbors
    _LOCAL_ITEM_NEIGHBORS = item_neighbors
    _LOCAL_EDGE_WEIGHTS = edge_weights

def score_candidate_chunk_task(task_args: Tuple[str, List[str], str]) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Worker task: Scores a partition/chunk of candidate items for a single target user.
    Args:
        task_args: (target_user, candidate_items_chunk, algorithm)
    Returns:
        List of (candidate_item, score, details)
    """
    target_user, candidate_items, algorithm = task_args
    results = []

    user_u_items = _LOCAL_USER_NEIGHBORS.get(target_user, set())
    if not user_u_items:
        return [(item, 0.0, {"algorithm": algorithm, "contributing_users_count": 0, "top_contributors": [], "raw_score": 0.0}) for item in candidate_items]

    for candidate_item in candidate_items:
        item_users = _LOCAL_ITEM_NEIGHBORS.get(candidate_item, set())

        if algorithm == "jaccard":
            score, details = score_candidate_jaccard(
                target_user=target_user,
                candidate_item=candidate_item,
                user_u_items=user_u_items,
                item_users=item_users,
                user_neighbors=_LOCAL_USER_NEIGHBORS,
                edge_weights=_LOCAL_EDGE_WEIGHTS
            )
        else: # Default to common_neighbors
            score, details = score_candidate_common_neighbors(
                target_user=target_user,
                candidate_item=candidate_item,
                user_u_items=user_u_items,
                item_users=item_users,
                user_neighbors=_LOCAL_USER_NEIGHBORS,
                edge_weights=_LOCAL_EDGE_WEIGHTS
            )

        results.append((candidate_item, score, details))

    return results

def batch_users_chunk_task(task_args: Tuple[List[str], List[str], str, int]) -> List[Tuple[str, List[Tuple[str, float, Dict[str, Any]]]]]:
    """
    Worker task: Scores recommendations for a batch partition of users.
    Args:
        task_args: (users_chunk, all_candidate_items, algorithm, top_k)
    Returns:
        List of (user_id, top_k_recommendations)
    """
    users_chunk, all_items_list, algorithm, top_k = task_args
    batch_results = []

    for user_id in users_chunk:
        user_u_items = _LOCAL_USER_NEIGHBORS.get(user_id, set())

        # Determine reachable candidates for this user
        candidates = set()
        for item in user_u_items:
            for other_user in _LOCAL_ITEM_NEIGHBORS.get(item, set()):
                if other_user != user_id:
                    for other_item in _LOCAL_USER_NEIGHBORS.get(other_user, set()):
                        if other_item not in user_u_items:
                            candidates.add(other_item)

        if not candidates:
            # Fall back to all unconsumed items
            candidates = set(all_items_list) - user_u_items

        candidate_scores = []
        for candidate_item in candidates:
            item_users = _LOCAL_ITEM_NEIGHBORS.get(candidate_item, set())
            if algorithm == "jaccard":
                score, details = score_candidate_jaccard(
                    target_user=user_id,
                    candidate_item=candidate_item,
                    user_u_items=user_u_items,
                    item_users=item_users,
                    user_neighbors=_LOCAL_USER_NEIGHBORS,
                    edge_weights=_LOCAL_EDGE_WEIGHTS
                )
            else:
                score, details = score_candidate_common_neighbors(
                    target_user=user_id,
                    candidate_item=candidate_item,
                    user_u_items=user_u_items,
                    item_users=item_users,
                    user_neighbors=_LOCAL_USER_NEIGHBORS,
                    edge_weights=_LOCAL_EDGE_WEIGHTS
                )
            candidate_scores.append((candidate_item, score, details))

        # Sort descending by score and pick top-k
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        batch_results.append((user_id, candidate_scores[:top_k]))

    return batch_results
