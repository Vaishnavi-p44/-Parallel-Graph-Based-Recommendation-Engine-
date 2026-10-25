"""
Quick verification script to test Graph Engine, Parallel Recommender, and Benchmarking.
"""

import os
from engine.graph_engine import UserItemGraph
from engine.parallel_recommender import ParallelRecommender
from engine.benchmark import BenchmarkSuite
from engine.visualizer import generate_benchmark_chart_base64

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    inter_csv = os.path.join(base_dir, "data", "movies_interactions.csv")
    meta_csv = os.path.join(base_dir, "data", "movies_metadata.csv")

    print("[1] Initializing Bipartite Graph...")
    graph = UserItemGraph(name="MoviesRecommender")
    graph.load_from_csv(inter_csv, meta_csv)
    stats = graph.get_graph_statistics()
    print(f"Graph loaded: {stats['num_users']} users, {stats['num_items']} items, {stats['num_edges']} edges, density: {stats['density_pct']}%")

    print("\n[2] Initializing Parallel Recommender...")
    recommender = ParallelRecommender(graph)

    target_user = "U001"
    print(f"\n[3] Generating recommendations for {target_user} (Sequential vs Parallel)...")

    # Sequential Common Neighbors
    res_seq = recommender.recommend_single_user(target_user, algorithm="common_neighbors", top_k=5, num_workers=1, parallel=False)
    print(f"Sequential Common Neighbors: {res_seq['execution_time_ms']} ms")
    for r in res_seq['recommendations']:
        print(f"  Rank {r['rank']}: {r['title']} (Score: {r['score']}) - {r['explanation'][:60]}...")

    # Parallel Common Neighbors
    res_par = recommender.recommend_single_user(target_user, algorithm="common_neighbors", top_k=5, num_workers=4, parallel=True)
    print(f"Parallel Common Neighbors (4 workers): {res_par['execution_time_ms']} ms")

    # Jaccard Similarity
    res_jaccard = recommender.recommend_single_user(target_user, algorithm="jaccard", top_k=5, num_workers=4, parallel=True)
    print(f"Parallel Jaccard (4 workers): {res_jaccard['execution_time_ms']} ms")
    for r in res_jaccard['recommendations']:
        print(f"  Rank {r['rank']}: {r['title']} (Score: {r['score']})")

    # Quick Benchmark
    print("\n[4] Running Benchmark Suite (Batch size 30 users, workers [1, 2, 4])...")
    suite = BenchmarkSuite(recommender)
    bench = suite.run_benchmark(algorithm="common_neighbors", user_batch_size=30, worker_counts=[1, 2, 4])
    for rec in bench["benchmark_results"]:
        print(f"  Workers: {rec['workers']} | Time: {rec['time_sec']}s | Speedup: {rec['speedup']}x | Efficiency: {rec['efficiency_pct']}%")

    # Matplotlib chart test
    chart_b64 = generate_benchmark_chart_base64(bench)
    print(f"\n[5] Benchmark Matplotlib chart generated: {len(chart_b64)} bytes base64 string.")
    print("\nAll engine modules verified successfully!")

if __name__ == "__main__":
    main()
