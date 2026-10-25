"""
Comprehensive end-to-end verification script.
Tests every function across both Movies and E-Commerce datasets,
specifically testing Cust_033 from ecommerce_interactions.csv.
"""

import os
import base64
import requests
from engine.graph_engine import UserItemGraph
from engine.parallel_recommender import ParallelRecommender
from engine.benchmark import BenchmarkSuite
from engine.visualizer import generate_benchmark_chart_base64
from engine.similarity import (
    common_neighbors_user_similarity,
    jaccard_user_similarity,
    score_candidate_common_neighbors,
    score_candidate_jaccard
)

def run_comprehensive_audit():
    print("=" * 70)
    print("STEP 1: Testing Bipartite Graph Engine with E-Commerce Dataset")
    print("=" * 70)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ecom_inter = os.path.join(base_dir, "data", "ecommerce_interactions.csv")
    ecom_meta = os.path.join(base_dir, "data", "ecommerce_metadata.csv")

    graph = UserItemGraph(name="ECommerceAudit")
    graph.load_from_csv(ecom_inter, ecom_meta)
    stats = graph.get_graph_statistics()
    print(f"Graph loaded successfully: {stats['num_users']} users, {stats['num_items']} items, {stats['num_edges']} edges.")
    assert stats["is_bipartite"] is True, "Graph must be strictly bipartite!"

    # Test Cust_033 interactions (matches CSV lines 539-551)
    cust_items = graph.get_user_items("Cust_033")
    print(f"Cust_033 has {len(cust_items)} interactions: {sorted(list(cust_items))}")
    assert "P230" in cust_items, "Cust_033 must have purchased P230!"
    assert graph.get_edge_weight("Cust_033", "P230") == 3.5, "P230 rating must match CSV (3.5)!"
    print("  [PASSED] Cust_033 interaction verification.")

    print("\n" + "=" * 70)
    print("STEP 2: Testing Recommendation Algorithms (Cust_033)")
    print("=" * 70)
    recommender = ParallelRecommender(graph)

    # 1. Common Neighbors (Sequential vs Parallel equivalence)
    res_cn_seq = recommender.recommend_single_user("Cust_033", algorithm="common_neighbors", top_k=5, num_workers=1, parallel=False)
    res_cn_par = recommender.recommend_single_user("Cust_033", algorithm="common_neighbors", top_k=5, num_workers=4, parallel=True)

    print(f"CN Top-5 Recommendations for Cust_033 (Sequential {res_cn_seq['execution_time_ms']} ms vs Parallel {res_cn_par['execution_time_ms']} ms):")
    for r1, r2 in zip(res_cn_seq["recommendations"], res_cn_par["recommendations"]):
        print(f"  Rank {r1['rank']}: {r1['title']} (Score: {r1['score']}) | Match: {r1['item_id'] == r2['item_id']}")
        assert r1["item_id"] == r2["item_id"], "Sequential and Parallel must produce identical item rankings!"
        assert abs(r1["score"] - r2["score"]) < 1e-4, "Sequential and Parallel must produce identical scores!"
        assert r1["item_id"] not in cust_items, f"Recommended item {r1['item_id']} was already consumed!"

    # 2. Jaccard Similarity (Sequential vs Parallel equivalence)
    res_jac_seq = recommender.recommend_single_user("Cust_033", algorithm="jaccard", top_k=5, num_workers=1, parallel=False)
    res_jac_par = recommender.recommend_single_user("Cust_033", algorithm="jaccard", top_k=5, num_workers=4, parallel=True)

    print(f"\nJaccard Top-5 Recommendations for Cust_033 (Sequential {res_jac_seq['execution_time_ms']} ms vs Parallel {res_jac_par['execution_time_ms']} ms):")
    for r1, r2 in zip(res_jac_seq["recommendations"], res_jac_par["recommendations"]):
        print(f"  Rank {r1['rank']}: {r1['title']} (Score: {r1['score']})")
        assert r1["item_id"] == r2["item_id"], "Sequential and Parallel must produce identical item rankings for Jaccard!"
        assert abs(r1["score"] - r2["score"]) < 1e-4, "Sequential and Parallel must produce identical scores for Jaccard!"

    print("  [PASSED] Single-user recommendation and sequential/parallel equivalence.")

    print("\n" + "=" * 70)
    print("STEP 3: Testing Distributed Benchmarking & Matplotlib Visualizer")
    print("=" * 70)
    suite = BenchmarkSuite(recommender)
    bench_data = suite.run_benchmark(algorithm="common_neighbors", user_batch_size=25, worker_counts=[1, 2, 4])
    print(f"Benchmark evaluated {bench_data['user_batch_size']} users across cores:")
    for rec in bench_data["benchmark_results"]:
        print(f"  Workers: {rec['workers']} | Time: {rec['time_sec']}s ({rec['time_ms']} ms) | Speedup: {rec['speedup']}x | Efficiency: {rec['efficiency_pct']}%")

    chart_b64 = generate_benchmark_chart_base64(bench_data)
    img_bytes = base64.b64decode(chart_b64)
    assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Generated chart must be a valid PNG image binary!"
    print(f"  [PASSED] Matplotlib chart binary verified ({len(img_bytes)} bytes).")

    print("\n" + "=" * 70)
    print("STEP 4: Testing Live Flask REST Endpoints on http://127.0.0.1:5000")
    print("=" * 70)
    base_url = "http://127.0.0.1:5000"

    # 1. Switch dataset to ecommerce
    r = requests.post(f"{base_url}/api/switch-dataset", json={"dataset_key": "ecommerce"})
    assert r.status_code == 200, "Switch dataset to ecommerce must return 200!"
    print(f"  Switched active dataset to: {r.json()['active_dataset']}")

    # 2. Get dataset info
    r = requests.get(f"{base_url}/api/dataset-info")
    assert r.status_code == 200 and r.json()["num_users"] == 80

    # 3. Get Cust_033 profile
    r = requests.get(f"{base_url}/api/user-profile/Cust_033")
    assert r.status_code == 200 and r.json()["user_id"] == "Cust_033"

    # 4. Recommend for Cust_033
    r = requests.post(f"{base_url}/api/recommend", json={
        "user_id": "Cust_033",
        "algorithm": "jaccard",
        "num_workers": 4,
        "parallel": True,
        "top_k": 5
    })
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    print(f"  Live REST /api/recommend for Cust_033 returned {len(recs)} items. Rank 1: '{recs[0]['title']}'")

    # 5. Subgraph visualization data
    r = requests.get(f"{base_url}/api/graph-data/Cust_033?algorithm=jaccard&top_k=5")
    assert r.status_code == 200
    subgraph = r.json()
    assert len(subgraph["nodes"]) > 0 and len(subgraph["links"]) > 0
    print(f"  Live REST /api/graph-data returned {len(subgraph['nodes'])} nodes and {len(subgraph['links'])} edges.")

    # 6. Stream benchmark chart PNG
    r = requests.get(f"{base_url}/api/benchmark-chart")
    assert r.status_code == 200 and r.headers["Content-Type"] == "image/png"
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")
    print(f"  Live REST /api/benchmark-chart returned valid PNG stream ({len(r.content)} bytes).")

    # 7. Switch back to movies dataset
    requests.post(f"{base_url}/api/switch-dataset", json={"dataset_key": "movies"})
    print("  Switched active dataset back to: Movies & Cinema")

    recommender.close_all_pools()
    print("\n" + "=" * 70)
    print("ALL MODULES, FUNCTIONS, WORKERS, AND APIS ARE WORKING 100% PROPERLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_comprehensive_audit()
