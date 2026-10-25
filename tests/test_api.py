"""
Unit tests for Flask Web API endpoints.
Tests REST responses, status codes, payload structures, and error handlers.
"""

import pytest
import json
from app import app, ACTIVE_RECOMMENDER

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
    if ACTIVE_RECOMMENDER:
        ACTIVE_RECOMMENDER.close_all_pools()

def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Parallel Graph Recommendation Engine" in res.data

def test_api_datasets(client):
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.get_json()
    assert "movies" in data["datasets"]
    assert "ecommerce" in data["datasets"]

def test_api_dataset_info(client):
    res = client.get("/api/dataset-info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["num_users"] > 0
    assert data["num_items"] > 0
    assert data["num_edges"] > 0
    assert data["is_bipartite"] is True

def test_api_users(client):
    res = client.get("/api/users")
    assert res.status_code == 200
    data = res.get_json()
    assert "users" in data
    assert len(data["users"]) > 0

def test_api_user_profile(client):
    res = client.get("/api/user-profile/U001")
    assert res.status_code == 200
    data = res.get_json()
    assert data["user_id"] == "U001"
    assert "history" in data
    assert "top_categories" in data

def test_api_recommend_success(client):
    payload = {
        "user_id": "U001",
        "algorithm": "common_neighbors",
        "num_workers": 2,
        "parallel": True,
        "top_k": 5
    }
    res = client.post("/api/recommend", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["user_id"] == "U001"
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["rank"] == 1
    assert "score" in data["recommendations"][0]
    assert "explanation" in data["recommendations"][0]

def test_api_recommend_unknown_user(client):
    payload = {
        "user_id": "NonExistentUser9999",
        "algorithm": "common_neighbors"
    }
    res = client.post("/api/recommend", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 404

def test_api_graph_data(client):
    res = client.get("/api/graph-data/U001?algorithm=common_neighbors&top_k=5")
    assert res.status_code == 200
    data = res.get_json()
    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) > 0
    assert len(data["links"]) > 0
