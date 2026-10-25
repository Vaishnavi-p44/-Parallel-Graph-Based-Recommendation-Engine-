# Parallel Graph-Based Recommendation Engine using Distributed Processing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Graph: NetworkX](https://img.shields.io/badge/Graph-NetworkX-green.svg)](https://networkx.org/)
[![Multiprocessing](https://img.shields.io/badge/Distributed-Multiprocessing-orange.svg)](https://docs.python.org/3/library/multiprocessing.html)
[![Tests: Pytest](https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg)](https://pytest.org/)

A distributed, high-performance graph-based recommendation system implemented in Python. The system represents user-item interactions as a bipartite graph and uses a **Master-Worker architecture** with Python's `multiprocessing` library to parallelize candidate-scoring tasks across multiple CPU cores, yielding real speedup and parallel efficiency.

Includes an interactive modern Web Dashboard built with Flask, Vanilla CSS, HTML5 Canvas, and Matplotlib.

---

## 🌟 Key Highlights

- **Bipartite Interaction Graph:** Built using NetworkX and indexed hash tables (`dict[str, set[str]]`) for $O(1)$ neighborhood lookups and verification of bipartiteness.
- **Graph-Based Filtering Algorithms:**
  - **Common Neighbors:** 3-hop bipartite path walk counting shared user interactions.
  - **Jaccard Similarity:** Normalized neighborhood overlap scoring between target user and candidate consumers.
- **Master-Worker Multiprocessing:**
  - Candidate item chunking for fine-grained parallel scoring.
  - User batch partitioning for production batch scoring and systematic benchmarking.
  - Process-local graph indexing via worker initializers to eliminate IPC serialization overhead.
  - 100% Windows-compatible with `spawn` protection using module-level workers and `if __name__ == '__main__':`.
- **Systematic Benchmarking Suite:**
  - Evaluates Sequential vs. Parallel execution across configurable core ladders (e.g. 1, 2, 4, 8 workers).
  - Computes Execution Time ($T$), Speedup ($S = T_1 / T_P$), Parallel Efficiency ($E = S / P$), and Throughput (users/sec).
  - Generates publication-grade multi-panel charts via Matplotlib.
- **Interactive Web Dashboard:**
  - Glassmorphic dark UI with modern typography (`Outfit` and `Inter`).
  - Interactive HTML5 Canvas bipartite network visualizer with force-directed physics, node dragging, and zoom/pan.
  - Real-time Top-5 recommendation cards with Rank, Score, Tags, and natural language explanations ("Why recommended?").
  - Dataset switcher (Movies vs. E-Commerce) and custom CSV upload support.
- **Comprehensive Unit Tests:** 21 automated tests covering graph topology, math correctness, parallel equivalence, and REST APIs.

---

## 📐 System Architecture

### Master-Worker Process Topology

```text
                        ┌───────────────────────────────┐
                        │         Flask Web App         │
                        │    (REST API & Dashboard)     │
                        └───────────────┬───────────────┘
                                        │ User Batch / Candidate Query
                                        ▼
                        ┌───────────────────────────────┐
                        │         Master Process        │
                        │    (Partitioning & Dispatch)  │
                        └───────┬───────────────┬───────┘
                                │               │
              ┌─────────────────┴──────┐ ┌──────┴─────────────────┐
              │                        │ │                        │
              ▼                        ▼ ▼                        ▼
       ┌──────────────┐         ┌──────────────┐           ┌──────────────┐
       │   Worker 1   │         │   Worker 2   │   ...     │   Worker P   │
       │ (Chunk 1)    │         │ (Chunk 2)    │           │ (Chunk P)    │
       └──────┬───────┘         └──────┬───────┘           └──────┬───────┘
              │                        │                          │
              │  Local Graph Index     │  Local Graph Index       │  Local Graph Index
              │  (Zero-Lock Read)      │  (Zero-Lock Read)        │  (Zero-Lock Read)
              │                        │                          │
              └─────────────────┬──────┴──────────┬───────────────┘
                                │                 │ Partial Scores
                                ▼                 ▼
                        ┌───────────────────────────────┐
                        │      Score Aggregation        │
                        │   (Global Sort & Top-5)       │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │  Dashboard / JSON Response    │
                        └───────────────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. Bipartite User-Item Graph
Let $G = (U \cup I, E)$ be an undirected bipartite graph where:
- $U = \{u_1, u_2, \dots, u_m\}$ is the set of User nodes (`bipartite=0`).
- $I = \{i_1, i_2, \dots, i_n\}$ is the set of Item nodes (`bipartite=1`).
- $E \subseteq U \times I$ represents observed interactions with edge weight $w(u, i)$.
- $N(u) \subseteq I$ denotes items consumed by user $u$.
- $N(i) \subseteq U$ denotes users who interacted with item $i$.

### 2. Common Neighbors (CN)
For a target user $u$ and an unconsumed candidate item $c \in I \setminus N(u)$:
$$\text{Score}_{CN}(u, c) = \sum_{v \in N(c) \setminus \{u\}} \left( |N(u) \cap N(v)| \times w(v, c) \right)$$

> **Mathematical Identity:** Counting shared neighbors in this bipartite projection is equivalent to counting all length-3 paths $u \to i \to v \to c$ from user $u$ to item $c$.

### 3. Jaccard Similarity
Normalizes neighborhood intersection by the total union of interacted items:
$$J(u, v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|} = \frac{|N(u) \cap N(v)|}{|N(u)| + |N(v)| - |N(u) \cap N(v)|}$$

Candidate item recommendation score:
$$\text{Score}_{Jaccard}(u, c) = \sum_{v \in N(c) \setminus \{u\}} \left( J(u, v) \times w(v, c) \right)$$

### 4. Parallel Speedup & Efficiency Metrics
- **Speedup Factor ($S_P$):**
  $$S_P = \frac{T_1}{T_P} = \frac{T_{\text{seq}}}{T_{\text{par}}(P)}$$
- **Parallel Efficiency ($E_P$):**
  $$E_P = \frac{S_P}{P} \times 100\% = \frac{T_1}{P \times T_P} \times 100\%$$
- **Amdahl's Law Theoretical Maximum:**
  $$S_{\max} = \frac{1}{(1 - f) + \frac{f}{P}}$$
  Where $f$ is the parallelizable fraction of the computation.

---

## 📂 Project Directory Structure

```text
PCproject/
├── app.py                     # Flask Web Application & REST API entry point
├── requirements.txt           # Python dependencies
├── generate_sample_data.py    # Generates rich synthetic datasets (Movies & E-Commerce)
├── test_quick_run.py          # Fast CLI verification of engine & benchmark
├── README.md                  # Complete technical documentation
├── data/
│   ├── movies_interactions.csv    # 2,700+ user-movie interactions
│   ├── movies_metadata.csv        # 96 curated movie titles, genres, ratings
│   ├── ecommerce_interactions.csv # 1,350+ customer-product interactions
│   ├── ecommerce_metadata.csv     # 60 products across audio, computing, gaming
│   └── uploads/                   # Storage for user-uploaded custom CSVs
├── engine/
│   ├── __init__.py
│   ├── graph_engine.py        # Bipartite UserItemGraph with NetworkX & adjacency sets
│   ├── similarity.py          # Pure Common Neighbors & Jaccard similarity functions
│   ├── workers.py             # Multiprocessing worker tasks & pool initializers
│   ├── parallel_recommender.py# Master-Worker orchestration & sequential baseline
│   ├── benchmark.py           # Systematic benchmarking suite & speedup metrics
│   └── visualizer.py          # Matplotlib multi-panel publication plot generator
├── static/
│   ├── css/
│   │   └── style.css          # Dark slate glassmorphism design system
│   └── js/
│       ├── app.js             # Dashboard frontend controller & API client
│       └── graph_canvas.js    # Interactive 2D HTML5 Canvas bipartite network simulation
├── templates/
│   └── index.html             # Single-page modern dashboard UI
└── tests/
    ├── __init__.py
    ├── test_graph_engine.py   # Bipartite structure, degree, and candidate tests
    ├── test_similarity.py     # Math unit tests for Common Neighbors & Jaccard
    ├── test_parallel_engine.py# Sequential vs parallel equivalence & benchmark tests
    └── test_api.py            # Flask REST API endpoints and response testing
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.14.0 Windows 64-bit)
- PowerShell or Command Prompt in Cursor IDE

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: (Optional) Re-generate Sample Data
Realistic sample datasets are pre-generated. To re-generate fresh datasets:
```powershell
python generate_sample_data.py
```

### Step 3: Run the Automated Unit Test Suite
Verify that all 21 unit tests pass:
```powershell
python -m pytest -v
```

### Step 4: Run the Quick CLI Verification
Test graph loading, sequential vs parallel execution, and benchmarking from the terminal:
```powershell
python test_quick_run.py
```

### Step 5: Start the Web Dashboard
Launch the Flask development server:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🌐 Web Dashboard Features

| Tab | Functionality |
| :--- | :--- |
| **Personalized Recommendations** | Select any target user, choose algorithm (Common Neighbors or Jaccard), select Sequential or Parallel mode with worker slider, view Top-5 recommendation cards with natural language explanations, and interact with the 2D Bipartite Subgraph Canvas. |
| **Distributed Benchmarking** | Select batch size (20 to 80 users) and worker configurations ($P \in [1, 2, 4, 8]$). View the live comparison table with real Execution Time, Speedup Factor, Parallel Efficiency, and download the high-resolution Matplotlib plot. |
| **Bipartite Graph Topology** | Real-time topological summary metrics: Total Users ($U$), Total Items ($I$), Edges ($E$), Graph Density, Degree distributions, and bipartite verification. |
| **Architecture & Mathematics** | Deep-dive explanations of master-worker orchestration, graph path formulations, Jaccard coefficients, and Amdahl's Law. |
| **Dataset Switcher / Upload** | Switch between Movies & Cinema and E-Commerce, or upload your own custom interaction and metadata CSV files directly from the UI. |

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/dataset-info` | `GET` | Returns graph statistics (users, items, edges, density, degrees). |
| `/api/datasets` | `GET` | Lists available datasets and the current selection. |
| `/api/switch-dataset` | `POST` | Switches active dataset (`{"dataset_key": "ecommerce"}`). |
| `/api/upload-dataset` | `POST` | Multipart form upload for custom `interactions.csv` and optional `metadata.csv`. |
| `/api/users` | `GET` | Returns list of users and interaction counts. |
| `/api/user-profile/<id>` | `GET` | Returns user interaction history and top genres. |
| `/api/recommend` | `POST` | Generates Top-$K$ recommendations (`{"user_id": "U001", "algorithm": "common_neighbors", "num_workers": 4, "parallel": true}`). |
| `/api/graph-data/<id>` | `GET` | Returns localized sub-graph nodes and links for interactive visualization. |
| `/api/benchmark` | `POST` | Executes benchmark across worker counts and returns timings, speedup, and chart. |
| `/api/benchmark-chart` | `GET` | Streams latest benchmark plot as a PNG image download. |

---

## 🛡️ Multi-Core Windows Compatibility

On Windows, Python uses the `spawn` multiprocessing start method (rather than `fork`). This project implements best practices for Windows multi-core execution:
1. All entry points are protected by `if __name__ == "__main__":` with `mp.freeze_support()`.
2. Worker functions (`score_candidate_chunk_task`, `batch_users_chunk_task`) are defined as picklable top-level functions in `engine/workers.py`.
3. Graph adjacency indices are populated inside worker processes using `mp.Pool(initializer=init_worker, ...)` to minimize inter-process serialization overhead.
4. Persistent pool caching eliminates redundant OS process creation during repeated queries.

---

## 📄 License
This project is open-source and available under the MIT License.
#   - P a r a l l e l - G r a p h - B a s e d - R e c o m m e n d a t i o n - E n g i n e -  
 