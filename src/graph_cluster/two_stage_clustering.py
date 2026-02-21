# -*- coding: utf-8 -*-
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

import networkx as nx
import igraph as ig
import leidenalg as la
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel


def load_graph_as_nx(path: str) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        raise ValueError("Invalid graph format")

    nodes_meta = []
    G = nx.Graph()

    for n in data["nodes"]:
        nid = n.get("id")
        nodes_meta.append(
            {
                "id": nid,
                "topic": n.get("topic", nid),
                "query": n.get("query", ""),
                "weight": n.get("weight", 0.0),
            }
        )
        G.add_node(nid)

    for e in data["edges"]:
        u = e.get("nodeA")
        v = e.get("nodeB")
        w = float(e.get("weight", 0.0))
        if u and v and u != v:
            G.add_edge(u, v, weight=w)

    return G, nodes_meta


def mean_pooling(output, mask):
    tok_emb = output.last_hidden_state
    mask = mask.unsqueeze(-1).expand(tok_emb.size()).float()
    return (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def get_all_embeddings(texts: List[str], model_path: str, batch_size=32) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
    except Exception:
        return np.array([])

    model.eval()
    all_embs = []

    cleaned_texts = [str(t).replace("\n", " ").strip() or "empty" for t in texts]

    with torch.no_grad():
        for i in tqdm(range(0, len(cleaned_texts), batch_size), ncols=100):
            batch = cleaned_texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = model(**inputs)
            emb = mean_pooling(out, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())

    if not all_embs:
        return np.array([])
    return torch.cat(all_embs, dim=0).numpy()


def run_leiden_weighted(G: nx.Graph, seed: int, resolution: float) -> Dict[str, int]:
    node_list = list(G.nodes())
    if not node_list:
        return {}

    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edges = []
    weights = []

    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(float(data.get("weight", 1.0)))

    ig_graph = ig.Graph(len(node_list), edges)
    ig_graph.es["weight"] = weights

    part = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        weights=ig_graph.es["weight"] if ig_graph.ecount() > 0 else None,
        seed=seed,
        resolution_parameter=resolution,
    )

    node2cluster = {}
    for cid, comm in enumerate(part):
        for idx in comm:
            node2cluster[node_list[idx]] = cid
    return node2cluster


def hierarchical_clustering_pipeline(G, nodes_meta, args):
    text_corpus = [n["query"] if n["query"] else n["topic"] for n in nodes_meta]
    embeddings_matrix = get_all_embeddings(text_corpus, args.model_path)

    if len(embeddings_matrix) == 0:
        return []

    id2emb_idx = {n["id"]: i for i, n in enumerate(nodes_meta)}

    node2super_cluster = run_leiden_weighted(
        G, seed=args.seed, resolution=args.leiden_resolution
    )

    super_clusters = {}
    for nid, sc_id in node2super_cluster.items():
        super_clusters.setdefault(sc_id, []).append(nid)

    TARGET_CLUSTER_SIZE = 7
    MIN_TOTAL_CLUSTERS = int(args.budget * 2)

    final_output = []
    global_sub_cluster_id = 0

    sc_k_allocations = {}
    total_k_planned = 0

    for sc_id, members in super_clusters.items():
        k = max(1, int(round(len(members) / TARGET_CLUSTER_SIZE)))
        sc_k_allocations[sc_id] = k
        total_k_planned += k

    if total_k_planned < MIN_TOTAL_CLUSTERS:
        scale_factor = MIN_TOTAL_CLUSTERS / total_k_planned
        for sc_id in sc_k_allocations:
            sc_k_allocations[sc_id] = max(
                1, int(sc_k_allocations[sc_id] * scale_factor)
            )

    for sc_id, members in super_clusters.items():
        n_sub_clusters = min(len(members), sc_k_allocations[sc_id])

        member_embeddings = []
        valid_ids = []
        for nid in members:
            idx = id2emb_idx.get(nid)
            if idx is not None:
                member_embeddings.append(embeddings_matrix[idx])
                valid_ids.append(nid)

        if not member_embeddings:
            continue

        member_embeddings = np.array(member_embeddings)

        kmeans = KMeans(
            n_clusters=n_sub_clusters, random_state=args.seed, n_init="auto"
        )
        sub_labels = kmeans.fit_predict(member_embeddings)

        groups = {i: [] for i in range(n_sub_clusters)}
        for i, label in enumerate(sub_labels):
            groups[label].append(valid_ids[i])

        for k_idx, sub_ids in groups.items():
            if not sub_ids:
                continue

            topics = [
                next((m for m in nodes_meta if m["id"] == nid), {})
                for nid in sub_ids
            ]

            final_output.append(
                {
                    "cluster_id": global_sub_cluster_id,
                    "super_cluster_id": sc_id,
                    "kmeans_sub_id": int(k_idx),
                    "nodes": sub_ids,
                    "topics": topics,
                    "size": len(sub_ids),
                }
            )
            global_sub_cluster_id += 1

    return final_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--stance", type=str, default="oppose", choices=["support", "oppose"])
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--model_path", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--leiden_resolution", type=float, default=0.8)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    topic_dir = os.path.join(args.base_dir, args.category, args.topic)
    input_path = os.path.join(topic_dir, "graph_{args.stance}.json")
    output_path = os.path.join(topic_dir, f"clusters_{args.stance}_{args.budget}.json")

    if not os.path.exists(input_path):
        return

    G, nodes_meta = load_graph_as_nx(input_path)
    if G.number_of_nodes() == 0:
        return

    final_clusters = hierarchical_clustering_pipeline(G, nodes_meta, args)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_clusters, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()