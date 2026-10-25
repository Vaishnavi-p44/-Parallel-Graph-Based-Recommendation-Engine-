"""
Bipartite User-Item Graph Engine.
Constructs, manages, and inspects bipartite graph structures using NetworkX and
high-performance adjacency mappings (dict-of-sets) for instant neighborhood lookups.
"""

from typing import Dict, Set, List, Tuple, Optional, Any
import os
import pandas as pd
import networkx as nx

class UserItemGraph:
    """
    High-performance Bipartite Graph representation for User-Item interactions.
    Maintains both NetworkX Graph instance (for topological metrics & visualizations)
    and indexed adjacency sets (for zero-overhead similarity computations).
    """

    def __init__(self, name: str = "UserItemGraph"):
        self.name = name
        self.graph = nx.Graph()

        # Adjacency indexes for O(1) set operations
        # user_neighbors[user_id] -> set of item_ids
        self.user_neighbors: Dict[str, Set[str]] = {}
        # item_neighbors[item_id] -> set of user_ids
        self.item_neighbors: Dict[str, Set[str]] = {}

        # Edge weights (e.g. ratings or interaction counts)
        # edge_weights[(user_id, item_id)] -> float
        self.edge_weights: Dict[Tuple[str, str], float] = {}

        # Item metadata: item_id -> dict of attributes (title, genre, price, etc.)
        self.item_metadata: Dict[str, Dict[str, Any]] = {}

        # User metadata / list
        self.all_users: Set[str] = set()
        self.all_items: Set[str] = set()

    def add_interaction(self, user_id: str, item_id: str, weight: float = 1.0, timestamp: Optional[int] = None):
        """Add an interaction edge between a user and an item."""
        user_id = str(user_id).strip()
        item_id = str(item_id).strip()
        weight = float(weight)

        self.all_users.add(user_id)
        self.all_items.add(item_id)

        # Update adjacency sets
        if user_id not in self.user_neighbors:
            self.user_neighbors[user_id] = set()
        self.user_neighbors[user_id].add(item_id)

        if item_id not in self.item_neighbors:
            self.item_neighbors[item_id] = set()
        self.item_neighbors[item_id].add(user_id)

        self.edge_weights[(user_id, item_id)] = weight

        # Update NetworkX graph
        if not self.graph.has_node(user_id):
            self.graph.add_node(user_id, bipartite=0, node_type="user")
        if not self.graph.has_node(item_id):
            self.graph.add_node(item_id, bipartite=1, node_type="item")

        self.graph.add_edge(user_id, item_id, weight=weight, timestamp=timestamp)

    def set_item_metadata(self, item_id: str, metadata: Dict[str, Any]):
        """Associate metadata with an item."""
        item_id = str(item_id).strip()
        self.item_metadata[item_id] = metadata
        if self.graph.has_node(item_id):
            for k, v in metadata.items():
                self.graph.nodes[item_id][k] = v

    def load_from_csv(self, interactions_csv_path: str, metadata_csv_path: Optional[str] = None):
        """
        Loads interactions and optional item metadata from CSV files.
        Expected interaction columns: user_id, item_id, [rating/weight], [timestamp]
        """
        if not os.path.exists(interactions_csv_path):
            raise FileNotFoundError(f"Interactions file not found: {interactions_csv_path}")

        df_inter = pd.read_csv(interactions_csv_path)

        # Standardize column names
        cols = {c.lower(): c for c in df_inter.columns}
        user_col = cols.get("user_id") or cols.get("user") or df_inter.columns[0]
        item_col = cols.get("item_id") or cols.get("item") or df_inter.columns[1]
        weight_col = cols.get("rating") or cols.get("weight") or cols.get("score")
        time_col = cols.get("timestamp") or cols.get("time")

        for _, row in df_inter.iterrows():
            uid = str(row[user_col])
            iid = str(row[item_col])
            w = float(row[weight_col]) if weight_col and pd.notna(row[weight_col]) else 1.0
            t = int(row[time_col]) if time_col and pd.notna(row[time_col]) else None
            self.add_interaction(uid, iid, weight=w, timestamp=t)

        # Load optional metadata
        if metadata_csv_path and os.path.exists(metadata_csv_path):
            df_meta = pd.read_csv(metadata_csv_path)
            meta_cols = {c.lower(): c for c in df_meta.columns}
            meta_item_col = meta_cols.get("item_id") or meta_cols.get("item") or df_meta.columns[0]

            for _, row in df_meta.iterrows():
                iid = str(row[meta_item_col])
                meta_dict = row.to_dict()
                self.set_item_metadata(iid, meta_dict)

    def get_user_items(self, user_id: str) -> Set[str]:
        """Returns the set of items interacted with by the specified user."""
        return self.user_neighbors.get(str(user_id), set())

    def get_item_users(self, item_id: str) -> Set[str]:
        """Returns the set of users who interacted with the specified item."""
        return self.item_neighbors.get(str(item_id), set())

    def get_edge_weight(self, user_id: str, item_id: str, default: float = 1.0) -> float:
        """Returns the weight of an interaction edge."""
        return self.edge_weights.get((str(user_id), str(item_id)), default)

    def get_candidate_items(self, user_id: str, reachable_only: bool = True) -> List[str]:
        """
        Find candidate items for a user (items the user hasn't consumed yet).
        If reachable_only is True, returns items connected via 2-hop user neighbors (3 hops from user).
        If no reachable candidates exist (cold user), returns all unconsumed items.
        """
        user_id = str(user_id)
        consumed = self.get_user_items(user_id)

        if not reachable_only:
            return sorted(list(self.all_items - consumed))

        candidates = set()
        for item in consumed:
            for other_user in self.item_neighbors.get(item, set()):
                if other_user != user_id:
                    for other_item in self.user_neighbors.get(other_user, set()):
                        if other_item not in consumed:
                            candidates.add(other_item)

        if not candidates:
            # Fall back to all unconsumed items
            candidates = self.all_items - consumed

        return sorted(list(candidates))

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Compute topological summary statistics of the bipartite graph."""
        num_users = len(self.all_users)
        num_items = len(self.all_items)
        num_edges = self.graph.number_of_edges()

        # Potential edges in bipartite graph = num_users * num_items
        max_bipartite_edges = num_users * num_items
        density = (num_edges / max_bipartite_edges) if max_bipartite_edges > 0 else 0.0

        user_degrees = [len(self.user_neighbors[u]) for u in self.all_users] if self.all_users else [0]
        item_degrees = [len(self.item_neighbors[i]) for i in self.all_items] if self.all_items else [0]

        return {
            "num_users": num_users,
            "num_items": num_items,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": num_edges,
            "density": round(density, 6),
            "density_pct": round(density * 100, 3),
            "avg_user_degree": round(sum(user_degrees) / len(user_degrees), 2) if user_degrees else 0,
            "max_user_degree": max(user_degrees) if user_degrees else 0,
            "min_user_degree": min(user_degrees) if user_degrees else 0,
            "avg_item_degree": round(sum(item_degrees) / len(item_degrees), 2) if item_degrees else 0,
            "max_item_degree": max(item_degrees) if item_degrees else 0,
            "min_item_degree": min(item_degrees) if item_degrees else 0,
            "is_bipartite": nx.is_bipartite(self.graph) if self.graph.number_of_nodes() > 0 else True
        }

    def get_subgraph_data(self, target_user: str, top_recommendations: Optional[List[str]] = None, max_neighbors: int = 6) -> Dict[str, Any]:
        """
        Extracts a localized sub-graph centered on target_user for interactive visualization:
        - Target user node
        - Sample of target user's interacted items
        - Top similar neighboring users who also interacted with those items
        - Recommended items
        - Edges connecting them
        """
        target_user = str(target_user)
        nodes = []
        links = []
        node_ids = set()

        # 1. Target User node
        nodes.append({
            "id": target_user,
            "label": f"Target User ({target_user})",
            "type": "target_user",
            "size": 22,
            "color": "#6366f1"
        })
        node_ids.add(target_user)

        user_items = list(self.get_user_items(target_user))[:max_neighbors]
        similar_users = set()

        # 2. Interacted Items
        for item in user_items:
            meta = self.item_metadata.get(item, {})
            title = meta.get("title", item)
            nodes.append({
                "id": item,
                "label": title,
                "type": "interacted_item",
                "size": 15,
                "color": "#10b981",
                "details": meta
            })
            node_ids.add(item)
            links.append({
                "source": target_user,
                "target": item,
                "weight": self.get_edge_weight(target_user, item),
                "type": "consumed"
            })

            # Collect a few similar users
            item_users = self.get_item_users(item)
            for u in item_users:
                if u != target_user and len(similar_users) < max_neighbors:
                    similar_users.add(u)

        # 3. Similar Users
        for su in similar_users:
            if su not in node_ids:
                nodes.append({
                    "id": su,
                    "label": f"User {su}",
                    "type": "similar_user",
                    "size": 14,
                    "color": "#f59e0b"
                })
                node_ids.add(su)

            # Connect similar user to shared interacted items
            for item in user_items:
                if item in self.get_user_items(su):
                    links.append({
                        "source": su,
                        "target": item,
                        "weight": self.get_edge_weight(su, item),
                        "type": "shared"
                    })

        # 4. Recommended Items
        if top_recommendations:
            for rec_item in top_recommendations:
                if rec_item not in node_ids:
                    meta = self.item_metadata.get(rec_item, {})
                    title = meta.get("title", rec_item)
                    nodes.append({
                        "id": rec_item,
                        "label": f"★ {title}",
                        "type": "recommended_item",
                        "size": 18,
                        "color": "#ec4899",
                        "details": meta
                    })
                    node_ids.add(rec_item)

                # Connect similar users to recommended items
                for su in similar_users:
                    if rec_item in self.get_user_items(su):
                        links.append({
                            "source": su,
                            "target": rec_item,
                            "weight": self.get_edge_weight(su, rec_item),
                            "type": "recommending"
                        })

                # Virtual recommendation link to target user (for visual clarity)
                links.append({
                    "source": target_user,
                    "target": rec_item,
                    "weight": 2.0,
                    "type": "predicted",
                    "dashed": True
                })

        return {
            "target_user": target_user,
            "nodes": nodes,
            "links": links
        }
