"""
Systematic Benchmarking Suite for Distributed Recommendation Engine.
Measures real performance across sequential and multi-core parallel configurations.
Computes:
- Execution Time (T)
- Speedup (S = T_seq / T_par)
- Parallel Efficiency (E = S / P)
- Throughput (Users or Items evaluated per second)
"""

import os
import platform
import multiprocessing as mp
from typing import List, Dict, Any, Optional

from engine.graph_engine import UserItemGraph
from engine.parallel_recommender import ParallelRecommender

class BenchmarkSuite:
    """
    Orchestrates systematic performance benchmarking comparing sequential and
    parallel worker execution across multiple core allocations.
    """

    def __init__(self, recommender: ParallelRecommender):
        self.recommender = recommender

    def run_benchmark(
        self,
        algorithm: str = "common_neighbors",
        user_batch_size: int = 50,
        worker_counts: Optional[List[int]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Executes benchmark over a batch of users across specified worker configurations.
        """
        all_users = list(self.recommender.graph.all_users)
        if not all_users:
            raise ValueError("Graph has no users to benchmark.")

        # Select test batch of users
        batch_size = min(user_batch_size, len(all_users))
        test_users = all_users[:batch_size]

        total_cores = mp.cpu_count()
        if not worker_counts:
            # Default scaling ladder tailored to system cores
            worker_counts = [1, 2, 4]
            if total_cores >= 8:
                worker_counts.append(8)
            if total_cores > 8 and total_cores not in worker_counts:
                worker_counts.append(total_cores)

        # Ensure 1 is always present for baseline
        if 1 not in worker_counts:
            worker_counts.insert(0, 1)
        worker_counts = sorted(list(set(worker_counts)))

        # Pre-warm all worker pools to separate OS spawn overhead from distributed compute throughput
        for p in worker_counts:
            if p > 1:
                self.recommender.get_or_create_pool(p)

        # 1. Sequential Baseline Execution (T_seq)
        seq_result = self.recommender.recommend_batch_users(
            user_ids=test_users,
            algorithm=algorithm,
            top_k=top_k,
            num_workers=1,
            parallel=False
        )
        t_seq = max(seq_result["execution_time_sec"], 0.0001)

        records: List[Dict[str, Any]] = []

        # Record sequential baseline
        records.append({
            "workers": 1,
            "mode": "sequential",
            "time_sec": round(t_seq, 5),
            "time_ms": round(t_seq * 1000, 2),
            "speedup": 1.0,
            "efficiency_pct": 100.0,
            "throughput_users_per_sec": round(batch_size / t_seq, 2)
        })

        # 2. Parallel Executions across worker counts
        for p in worker_counts:
            if p == 1:
                continue

            par_result = self.recommender.recommend_batch_users(
                user_ids=test_users,
                algorithm=algorithm,
                top_k=top_k,
                num_workers=p,
                parallel=True,
                use_persistent_pool=True
            )
            t_par = max(par_result["execution_time_sec"], 0.0001)

            speedup = t_seq / t_par
            efficiency = (speedup / p) * 100.0
            throughput = batch_size / t_par

            records.append({
                "workers": p,
                "mode": "parallel",
                "time_sec": round(t_par, 5),
                "time_ms": round(t_par * 1000, 2),
                "speedup": round(speedup, 3),
                "efficiency_pct": round(efficiency, 2),
                "throughput_users_per_sec": round(throughput, 2)
            })

        return {
            "algorithm": algorithm,
            "user_batch_size": batch_size,
            "total_system_cores": total_cores,
            "system_info": {
                "os": platform.system(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "sequential_baseline_sec": round(t_seq, 5),
            "benchmark_results": records
        }
