"""
Unit tests for Multiprocessing Master-Worker Parallel Recommender and Benchmarking.
Verifies equivalence between sequential and distributed parallel computations.
"""

import os
import pytest
from engine.graph_engine import UserItemGraph
from engine.parallel_recommender import ParallelRecommender
from engine.benchmark import BenchmarkSuite

@pytest.fixture(scope="module")
def populated_graph():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inter_csv = os.path.join(base_dir, "data", "movies_interactions.csv")
    meta_csv = os.path.join(base_dir, "data", "movies_metadata.csv")
    graph = UserItemGraph(name="TestMovies")
    graph.load_from_csv(inter_csv, meta_csv)
    return graph

@pytest.fixture(scope="module")
def recommender(populated_graph):
    rec = ParallelRecommender(populated_graph)
    yield rec
    rec.close_all_pools()

def test_single_user_sequential_vs_parallel_equivalence(recommender):
    target_user = "U001"

    # 1. Sequential run
    seq_result = recommender.recommend_single_user(
        target_user=target_user,
        algorithm="common_neighbors",
        top_k=5,
        num_workers=1,
        parallel=False
    )

    # 2. Parallel run with 2 workers
    par_result = recommender.recommend_single_user(
        target_user=target_user,
        algorithm="common_neighbors",
        top_k=5,
        num_workers=2,
        parallel=True
    )

    assert seq_result["candidates_evaluated"] == par_result["candidates_evaluated"]
    assert len(seq_result["recommendations"]) == len(par_result["recommendations"])

    # Verify identical recommendations and identical scores
    for seq_rec, par_rec in zip(seq_result["recommendations"], par_result["recommendations"]):
        assert seq_rec["item_id"] == par_rec["item_id"]
        assert seq_rec["score"] == pytest.approx(par_rec["score"], rel=1e-4)

def test_jaccard_sequential_vs_parallel_equivalence(recommender):
    target_user = "U002"

    seq_result = recommender.recommend_single_user(
        target_user=target_user,
        algorithm="jaccard",
        top_k=5,
        num_workers=1,
        parallel=False
    )

    par_result = recommender.recommend_single_user(
        target_user=target_user,
        algorithm="jaccard",
        top_k=5,
        num_workers=2,
        parallel=True
    )

    for seq_rec, par_rec in zip(seq_result["recommendations"], par_result["recommendations"]):
        assert seq_rec["item_id"] == par_rec["item_id"]
        assert seq_rec["score"] == pytest.approx(par_rec["score"], rel=1e-4)

def test_batch_users_equivalence(recommender):
    batch_users = ["U001", "U002", "U003", "U004"]

    seq_batch = recommender.recommend_batch_users(
        user_ids=batch_users,
        algorithm="common_neighbors",
        top_k=3,
        num_workers=1,
        parallel=False
    )

    par_batch = recommender.recommend_batch_users(
        user_ids=batch_users,
        algorithm="common_neighbors",
        top_k=3,
        num_workers=2,
        parallel=True
    )

    assert seq_batch["num_users"] == par_batch["num_users"]
    assert len(seq_batch["results_sample"]) == len(par_batch["results_sample"])

def test_benchmark_suite_metrics(recommender):
    suite = BenchmarkSuite(recommender)
    bench_data = suite.run_benchmark(
        algorithm="common_neighbors",
        user_batch_size=15,
        worker_counts=[1, 2],
        top_k=3
    )

    assert "benchmark_results" in bench_data
    results = bench_data["benchmark_results"]
    assert len(results) >= 2

    # Verify baseline record
    baseline = results[0]
    assert baseline["workers"] == 1
    assert baseline["speedup"] == 1.0
    assert baseline["efficiency_pct"] == 100.0

    # Verify parallel record
    par_rec = results[1]
    assert par_rec["workers"] == 2
    assert par_rec["speedup"] > 0
    assert par_rec["efficiency_pct"] > 0
    assert par_rec["throughput_users_per_sec"] > 0
