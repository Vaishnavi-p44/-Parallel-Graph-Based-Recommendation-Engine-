"""
Flask Web Application & REST API Server for
Parallel Graph-Based Recommendation Engine using Distributed Processing.
"""

import os
import io
import base64
import multiprocessing as mp
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from engine.graph_engine import UserItemGraph
from engine.parallel_recommender import ParallelRecommender
from engine.benchmark import BenchmarkSuite
from engine.visualizer import generate_benchmark_chart_base64

app = Flask(__name__)
app.config["SECRET_KEY"] = "distributed-recsys-secret-key-2026"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Available Datasets
DATASETS = {
    "movies": {
        "name": "Movies & Cinema (MovieLens Style)",
        "description": "2,700+ interactions across 100 users and 96 movies with genres and ratings.",
        "interactions": os.path.join(DATA_DIR, "movies_interactions.csv"),
        "metadata": os.path.join(DATA_DIR, "movies_metadata.csv"),
        "type": "Entertainment"
    },
    "ecommerce": {
        "name": "E-Commerce Tech Gadgets",
        "description": "1,350+ interactions across 80 customers and 60 tech products with prices and categories.",
        "interactions": os.path.join(DATA_DIR, "ecommerce_interactions.csv"),
        "metadata": os.path.join(DATA_DIR, "ecommerce_metadata.csv"),
        "type": "Retail"
    }
}

CURRENT_DATASET_KEY = "movies"
ACTIVE_GRAPH: Optional[UserItemGraph] = None
ACTIVE_RECOMMENDER: Optional[ParallelRecommender] = None
LATEST_BENCHMARK_DATA: Optional[Dict[str, Any]] = None

def load_engine_dataset(dataset_key: str):
    """Loads and initializes the graph engine for the specified dataset."""
    global CURRENT_DATASET_KEY, ACTIVE_GRAPH, ACTIVE_RECOMMENDER
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    cfg = DATASETS[dataset_key]
    graph = UserItemGraph(name=cfg["name"])
    graph.load_from_csv(cfg["interactions"], cfg["metadata"])

    # Clean up previous pools if any
    if ACTIVE_RECOMMENDER:
        ACTIVE_RECOMMENDER.close_all_pools()

    CURRENT_DATASET_KEY = dataset_key
    ACTIVE_GRAPH = graph
    ACTIVE_RECOMMENDER = ParallelRecommender(graph)
    print(f"Successfully loaded dataset: {cfg['name']}")

# Initialize default dataset on module load
load_engine_dataset("movies")


@app.route("/")
def index():
    """Renders the main recommendation engine dashboard."""
    return render_template("index.html")


@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """Lists available datasets and active selection."""
    return jsonify({
        "active_dataset": CURRENT_DATASET_KEY,
        "datasets": {
            k: {
                "name": v["name"],
                "description": v["description"],
                "type": v["type"]
            }
            for k, v in DATASETS.items()
        }
    })


@app.route("/api/switch-dataset", methods=["POST"])
def switch_dataset():
    """Switches the active dataset."""
    data = request.get_json() or {}
    dataset_key = data.get("dataset_key")

    if not dataset_key or dataset_key not in DATASETS:
        return jsonify({"error": f"Invalid dataset key: {dataset_key}"}), 400

    try:
        load_engine_dataset(dataset_key)
        stats = ACTIVE_GRAPH.get_graph_statistics()
        return jsonify({
            "message": f"Switched to {DATASETS[dataset_key]['name']}",
            "active_dataset": dataset_key,
            "statistics": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-dataset", methods=["POST"])
def upload_dataset():
    """Uploads a custom interaction CSV and optional metadata CSV."""
    if "interactions_file" not in request.files:
        return jsonify({"error": "No interactions_file provided."}), 400

    inter_file = request.files["interactions_file"]
    meta_file = request.files.get("metadata_file")
    dataset_name = request.form.get("dataset_name", "Custom Dataset").strip()

    if inter_file.filename == "":
        return jsonify({"error": "Empty filename provided."}), 400

    inter_filename = secure_filename(f"custom_inter_{inter_file.filename}")
    inter_path = os.path.join(app.config["UPLOAD_FOLDER"], inter_filename)
    inter_file.save(inter_path)

    meta_path = None
    if meta_file and meta_file.filename:
        meta_filename = secure_filename(f"custom_meta_{meta_file.filename}")
        meta_path = os.path.join(app.config["UPLOAD_FOLDER"], meta_filename)
        meta_file.save(meta_path)

    custom_key = f"custom_{len(DATASETS) + 1}"
    DATASETS[custom_key] = {
        "name": dataset_name,
        "description": f"Custom uploaded dataset with {dataset_name}.",
        "interactions": inter_path,
        "metadata": meta_path,
        "type": "Custom"
    }

    try:
        load_engine_dataset(custom_key)
        stats = ACTIVE_GRAPH.get_graph_statistics()
        return jsonify({
            "message": f"Successfully uploaded and activated {dataset_name}!",
            "active_dataset": custom_key,
            "statistics": stats
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse custom CSV: {str(e)}"}), 400


@app.route("/api/dataset-info", methods=["GET"])
def dataset_info():
    """Returns graph topological statistics and system resource metrics."""
    if not ACTIVE_GRAPH:
        return jsonify({"error": "No active graph loaded."}), 500

    stats = ACTIVE_GRAPH.get_graph_statistics()
    stats["dataset_name"] = DATASETS[CURRENT_DATASET_KEY]["name"]
    stats["dataset_description"] = DATASETS[CURRENT_DATASET_KEY]["description"]
    stats["cpu_cores_available"] = mp.cpu_count()
    return jsonify(stats)


@app.route("/api/users", methods=["GET"])
def get_users():
    """Returns list of users in the active dataset with summary metrics."""
    if not ACTIVE_GRAPH:
        return jsonify({"error": "No active graph loaded."}), 500

    user_list = []
    # Sample or list all users
    for uid in sorted(list(ACTIVE_GRAPH.all_users)):
        items = ACTIVE_GRAPH.get_user_items(uid)
        user_list.append({
            "user_id": uid,
            "interaction_count": len(items)
        })

    return jsonify({"users": user_list, "total_users": len(user_list)})


@app.route("/api/user-profile/<user_id>", methods=["GET"])
def get_user_profile(user_id: str):
    """Returns profile and interaction history for a given user."""
    if not ACTIVE_GRAPH:
        return jsonify({"error": "No active graph loaded."}), 500

    user_id = str(user_id).strip()
    if user_id not in ACTIVE_GRAPH.all_users:
        return jsonify({"error": f"User {user_id} not found."}), 404

    items = ACTIVE_GRAPH.get_user_items(user_id)
    history = []
    category_counts: Dict[str, int] = {}

    for item_id in items:
        meta = ACTIVE_GRAPH.item_metadata.get(item_id, {})
        rating = ACTIVE_GRAPH.get_edge_weight(user_id, item_id)
        cat = meta.get("genres") or meta.get("category", "General")

        for c in str(cat).split(","):
            c_clean = c.strip()
            category_counts[c_clean] = category_counts.get(c_clean, 0) + 1

        history.append({
            "item_id": item_id,
            "title": meta.get("title", f"Item {item_id}"),
            "category": cat,
            "rating": rating,
            "metadata": meta
        })

    history.sort(key=lambda x: x["rating"], reverse=True)
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:4]

    return jsonify({
        "user_id": user_id,
        "total_interactions": len(items),
        "top_categories": [{"name": k, "count": v} for k, v in top_categories],
        "history": history
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Generates Top-K personalized recommendations for a user.
    Supports Common Neighbors and Jaccard Similarity,
    Sequential vs Master-Worker Parallel execution with configurable worker count.
    """
    if not ACTIVE_RECOMMENDER:
        return jsonify({"error": "Engine not initialized."}), 500

    data = request.get_json() or {}
    user_id = data.get("user_id")
    algorithm = data.get("algorithm", "common_neighbors")
    num_workers = int(data.get("num_workers", 4))
    parallel = bool(data.get("parallel", True))
    top_k = int(data.get("top_k", 5))

    if not user_id:
        return jsonify({"error": "user_id is required."}), 400

    if user_id not in ACTIVE_GRAPH.all_users:
        return jsonify({"error": f"User '{user_id}' does not exist in dataset."}), 404

    try:
        result = ACTIVE_RECOMMENDER.recommend_single_user(
            target_user=user_id,
            algorithm=algorithm,
            top_k=top_k,
            num_workers=num_workers,
            parallel=parallel
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph-data/<user_id>", methods=["GET"])
def get_graph_data(user_id: str):
    """
    Returns localized bipartite sub-graph nodes and links for interactive visualization:
    Target User -> Interacted Items -> Similar Users -> Recommended Items.
    """
    if not ACTIVE_GRAPH or not ACTIVE_RECOMMENDER:
        return jsonify({"error": "Engine not initialized."}), 500

    user_id = str(user_id).strip()
    if user_id not in ACTIVE_GRAPH.all_users:
        return jsonify({"error": f"User {user_id} not found."}), 404

    algorithm = request.args.get("algorithm", "common_neighbors")
    top_k = int(request.args.get("top_k", 5))

    try:
        # Get top recommended item IDs to highlight them in the sub-graph
        rec_res = ACTIVE_RECOMMENDER.recommend_single_user(
            target_user=user_id,
            algorithm=algorithm,
            top_k=top_k,
            num_workers=2,
            parallel=True
        )
        rec_item_ids = [r["item_id"] for r in rec_res.get("recommendations", [])]

        subgraph = ACTIVE_GRAPH.get_subgraph_data(
            target_user=user_id,
            top_recommendations=rec_item_ids,
            max_neighbors=6
        )
        return jsonify(subgraph)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """
    Executes systematic sequential vs parallel benchmark across worker configurations.
    Returns timings, speedup S(P), parallel efficiency E(P), and base64 Matplotlib chart.
    """
    global LATEST_BENCHMARK_DATA
    if not ACTIVE_RECOMMENDER:
        return jsonify({"error": "Engine not initialized."}), 500

    data = request.get_json() or {}
    algorithm = data.get("algorithm", "common_neighbors")
    batch_size = int(data.get("user_batch_size", 40))
    worker_counts = data.get("worker_counts") or [1, 2, 4, 8]
    top_k = int(data.get("top_k", 5))

    # Cap worker counts to machine cores + 2
    max_cores = mp.cpu_count()
    worker_counts = [w for w in worker_counts if w <= max_cores]
    if not worker_counts:
        worker_counts = [1, 2, 4]
    if 1 not in worker_counts:
        worker_counts.insert(0, 1)

    try:
        suite = BenchmarkSuite(ACTIVE_RECOMMENDER)
        bench_result = suite.run_benchmark(
            algorithm=algorithm,
            user_batch_size=batch_size,
            worker_counts=worker_counts,
            top_k=top_k
        )

        # Generate publication-grade Matplotlib chart
        chart_b64 = generate_benchmark_chart_base64(bench_result)
        bench_result["chart_base64"] = chart_b64
        LATEST_BENCHMARK_DATA = bench_result

        return jsonify(bench_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark-chart", methods=["GET"])
def get_benchmark_chart():
    """Returns the latest benchmark chart as a PNG image."""
    global LATEST_BENCHMARK_DATA
    if not LATEST_BENCHMARK_DATA or "chart_base64" not in LATEST_BENCHMARK_DATA:
        # Run a quick default benchmark to generate chart
        suite = BenchmarkSuite(ACTIVE_RECOMMENDER)
        bench_result = suite.run_benchmark(algorithm="common_neighbors", user_batch_size=20, worker_counts=[1, 2, 4])
        chart_b64 = generate_benchmark_chart_base64(bench_result)
        LATEST_BENCHMARK_DATA = bench_result
        LATEST_BENCHMARK_DATA["chart_base64"] = chart_b64

    img_data = base64.b64decode(LATEST_BENCHMARK_DATA["chart_base64"])
    return send_file(io.BytesIO(img_data), mimetype="image/png", download_name="benchmark_scaling_chart.png")


if __name__ == "__main__":
    # Ensure Windows multiprocessing compatibility
    mp.freeze_support()
    print("Starting Parallel Graph-Based Recommendation Engine on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
