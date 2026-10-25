"""
Master-Worker Distributed Recommendation Engine.
Orchestrates candidate scoring across multiple worker processes using multiprocessing.
Supports:
1. Candidate Item Chunking (fine-grained single-user master-worker distribution)
2. Batch User Chunking (coarse-grained multi-user master-worker distribution)
3. Sequential baseline execution for comparative benchmarking
"""

import time
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Optional

from engine.graph_engine import UserItemGraph
from engine.similarity import score_candidate_common_neighbors, score_candidate_jaccard
from engine.workers import init_worker, score_candidate_chunk_task, batch_users_chunk_task

class ParallelRecommender:
    """
    Distributed Recommendation Engine utilizing Master-Worker process topology.
    Supports warm persistent pools to eliminate repetitive process spawn overhead on Windows.
    """

    def __init__(self, graph: UserItemGraph):
        self.graph = graph
        self._pools: Dict[int, mp.Pool] = {}

    def get_or_create_pool(self, num_workers: int) -> mp.Pool:
        """Retrieves or initializes a warm worker pool of size num_workers."""
        if num_workers not in self._pools:
            pool = mp.Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(
                    self.graph.user_neighbors,
                    self.graph.item_neighbors,
                    self.graph.edge_weights
                )
            )
            # Send lightweight dummy task to finish OS process spawning handshake
            try:
                dummy_tasks = [("warmup_user", [], "common_neighbors")] * num_workers
                pool.map(score_candidate_chunk_task, dummy_tasks)
            except Exception:
                pass

            self._pools[num_workers] = pool
        return self._pools[num_workers]

    def close_all_pools(self):
        """Closes and terminates all cached worker pools."""
        for pool in self._pools.values():
            try:
                pool.close()
                pool.join()
            except Exception:
                pass
        self._pools.clear()

    def _sequential_score_candidates(
        self,
        target_user: str,
        candidates: List[str],
        algorithm: str
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Runs candidate scoring sequentially in the current process."""
        user_items = self.graph.get_user_items(target_user)
        results = []

        for candidate in candidates:
            item_users = self.graph.get_item_users(candidate)
            if algorithm == "jaccard":
                score, details = score_candidate_jaccard(
                    target_user=target_user,
                    candidate_item=candidate,
                    user_u_items=user_items,
                    item_users=item_users,
                    user_neighbors=self.graph.user_neighbors,
                    edge_weights=self.graph.edge_weights
                )
            else:
                score, details = score_candidate_common_neighbors(
                    target_user=target_user,
                    candidate_item=candidate,
                    user_u_items=user_items,
                    item_users=item_users,
                    user_neighbors=self.graph.user_neighbors,
                    edge_weights=self.graph.edge_weights
                )
            results.append((candidate, score, details))

        return results

    def recommend_single_user(
        self,
        target_user: str,
        algorithm: str = "common_neighbors",
        top_k: int = 5,
        num_workers: int = 4,
        parallel: bool = True,
        use_persistent_pool: bool = True
    ) -> Dict[str, Any]:
        """
        Generates Top-K recommendations for a single user.
        If parallel=True and num_workers > 1, candidates are partitioned across worker processes.
        """
        target_user = str(target_user).strip()
        if target_user not in self.graph.all_users:
            raise ValueError(f"User '{target_user}' does not exist in graph.")

        # 1. Candidate Generation
        candidates = self.graph.get_candidate_items(target_user, reachable_only=True)
        if not candidates:
            candidates = self.graph.get_candidate_items(target_user, reachable_only=False)

        num_candidates = len(candidates)
        scored_items: List[Tuple[str, float, Dict[str, Any]]] = []

        if not parallel or num_workers <= 1 or num_candidates < 2:
            mode = "sequential"
            actual_workers = 1
            start_time = time.perf_counter()
            scored_items = self._sequential_score_candidates(target_user, candidates, algorithm)
            exec_time = time.perf_counter() - start_time
        else:
            mode = "parallel"
            actual_workers = min(num_workers, num_candidates, mp.cpu_count())

            # Partition candidate items into balanced chunks
            chunks: List[List[str]] = [[] for _ in range(actual_workers)]
            for idx, item in enumerate(candidates):
                chunks[idx % actual_workers].append(item)

            tasks = [(target_user, chunk, algorithm) for chunk in chunks if chunk]

            if use_persistent_pool:
                pool = self.get_or_create_pool(actual_workers)
                start_time = time.perf_counter()
                chunk_results = pool.map(score_candidate_chunk_task, tasks)
                exec_time = time.perf_counter() - start_time
            else:
                start_time = time.perf_counter()
                with mp.Pool(
                    processes=len(tasks),
                    initializer=init_worker,
                    initargs=(
                        self.graph.user_neighbors,
                        self.graph.item_neighbors,
                        self.graph.edge_weights
                    )
                ) as pool:
                    chunk_results = pool.map(score_candidate_chunk_task, tasks)
                exec_time = time.perf_counter() - start_time

            for worker_result in chunk_results:
                scored_items.extend(worker_result)

        # Master sorts candidate items descending by similarity score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:top_k]

        # Format Top-K recommendation payloads with rich metadata
        recommendations = []
        for rank, (item_id, score, details) in enumerate(top_items, start=1):
            meta = self.graph.item_metadata.get(item_id, {})
            title = meta.get("title", f"Item {item_id}")

            contrib_count = details.get("contributing_users_count", 0)
            top_contrib = details.get("top_contributors", [])

            if contrib_count > 0:
                top_u = top_contrib[0]["user_id"]
                top_sim = top_contrib[0]["similarity"]
                why = (
                    f"Recommended because {contrib_count} similar user(s) engaged with it. "
                    f"Most notably User {top_u} who shares {top_contrib[0].get('shared_count', 1)} common items "
                    f"with you ({algorithm.replace('_', ' ').title()} similarity: {top_sim})."
                )
            else:
                why = "Recommended based on overall network connectivity."

            recommendations.append({
                "rank": rank,
                "item_id": item_id,
                "title": title,
                "score": round(score, 4),
                "details": details,
                "metadata": meta,
                "explanation": why
            })

        return {
            "user_id": target_user,
            "algorithm": algorithm,
            "mode": mode,
            "num_workers": actual_workers,
            "candidates_evaluated": num_candidates,
            "execution_time_sec": round(exec_time, 5),
            "execution_time_ms": round(exec_time * 1000, 2),
            "recommendations": recommendations
        }

    def recommend_batch_users(
        self,
        user_ids: List[str],
        algorithm: str = "common_neighbors",
        top_k: int = 5,
        num_workers: int = 4,
        parallel: bool = True,
        use_persistent_pool: bool = True
    ) -> Dict[str, Any]:
        """
        Executes recommendations for a batch of users (used for high-throughput batch evaluation & benchmarking).
        """
        valid_users = [u for u in user_ids if u in self.graph.all_users]
        if not valid_users:
            raise ValueError("No valid users provided in batch.")

        all_items_list = list(self.graph.all_items)
        num_users = len(valid_users)
        batch_results: List[Tuple[str, List[Tuple[str, float, Dict[str, Any]]]]] = []

        if not parallel or num_workers <= 1 or num_users < 2:
            mode = "sequential"
            actual_workers = 1
            task_args = (valid_users, all_items_list, algorithm, top_k)
            init_worker(self.graph.user_neighbors, self.graph.item_neighbors, self.graph.edge_weights)
            start_time = time.perf_counter()
            batch_results = batch_users_chunk_task(task_args)
            exec_time = time.perf_counter() - start_time
        else:
            mode = "parallel"
            actual_workers = min(num_workers, num_users, mp.cpu_count())

            # Partition users across workers
            user_chunks: List[List[str]] = [[] for _ in range(actual_workers)]
            for idx, u in enumerate(valid_users):
                user_chunks[idx % actual_workers].append(u)

            tasks = [(chunk, all_items_list, algorithm, top_k) for chunk in user_chunks if chunk]

            if use_persistent_pool:
                pool = self.get_or_create_pool(actual_workers)
                start_time = time.perf_counter()
                chunk_outputs = pool.map(batch_users_chunk_task, tasks)
                exec_time = time.perf_counter() - start_time
            else:
                start_time = time.perf_counter()
                with mp.Pool(
                    processes=len(tasks),
                    initializer=init_worker,
                    initargs=(
                        self.graph.user_neighbors,
                        self.graph.item_neighbors,
                        self.graph.edge_weights
                    )
                ) as pool:
                    chunk_outputs = pool.map(batch_users_chunk_task, tasks)
                exec_time = time.perf_counter() - start_time

            for worker_output in chunk_outputs:
                batch_results.extend(worker_output)

        throughput = (num_users / exec_time) if exec_time > 0 else 0.0

        return {
            "num_users": num_users,
            "algorithm": algorithm,
            "mode": mode,
            "num_workers": actual_workers,
            "execution_time_sec": round(exec_time, 5),
            "execution_time_ms": round(exec_time * 1000, 2),
            "throughput_users_per_sec": round(throughput, 2),
            "results_sample": [
                {
                    "user_id": u,
                    "top_recommendations": [
                        {"item_id": it, "score": round(sc, 4), "title": self.graph.item_metadata.get(it, {}).get("title", it)}
                        for it, sc, det in recs[:3]
                    ]
                }
                for u, recs in batch_results[:5]
            ]
        }
