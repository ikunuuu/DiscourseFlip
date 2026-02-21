import os
import json
import argparse
import re
import math
from typing import List, Dict, Any, Tuple, Set
import networkx as nx  

def edge_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([a, b]))

def normalize_doc(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def rank_discount(rank: int) -> float:
    return 1.0 / (math.log2(2.0 + max(1, rank)))

def filter_raw_data_by_stance(
    raw_data: List[Dict[str, Any]], 
    clean_json_path: str, 
    target_stance: str
) -> List[Dict[str, Any]]:
    print(f"[Filter] Loading polarity scores from: {clean_json_path}")
    
    if not os.path.exists(clean_json_path):
        print(f"[WARN] Clean JSON not found at {clean_json_path}. Skipping filtering (keeping all nodes).")
        return raw_data

    try:
        with open(clean_json_path, 'r', encoding='utf-8') as f:
            clean_data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load clean json: {e}. Skipping filtering.")
        return raw_data
        
    query_score_map = {item.get('query', '').strip(): float(item.get('asv_score', 1.0)) for item in clean_data}

    banned_scores = set()
    if target_stance == "support":
        banned_scores = {2.0}
    elif target_stance == "oppose":
        banned_scores = {0.0}
    else:
        print(f"[WARN] Unknown stance '{target_stance}'. No filtering applied.")
        return raw_data

    filtered_data = []
    removed_count = 0
    kept_count = 0

    for item in raw_data:
        q = item.get('query', '').strip()
        score = query_score_map.get(q, 1.0)
        if score in banned_scores:
            removed_count += 1
        else:
            filtered_data.append(item)
            kept_count += 1

    print(f"[Polarity Filter] Target Stance: {target_stance}")
    print(f"  - Total Queries : {len(raw_data)}")
    print(f"  - Removed Nodes : {removed_count} (Already aligned)")
    print(f"  - Kept Nodes    : {kept_count} (Valid targets)")
    
    return filtered_data


def build_nodes(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = []
    for idx, item in enumerate(raw_data, start=1):
        raw_docs = item.get("top_k_docs", []) or []
        seen = set()
        docs_ordered = []
        for d in raw_docs:
            if not isinstance(d, str) or not d.strip():
                continue
            nd = normalize_doc(d)
            if nd in seen:
                continue
            seen.add(nd)
            docs_ordered.append(nd)
        nid = item.get("id") or f"N{idx}"
        nodes.append(
            {
                "id": nid,
                "topic": item.get("topic", f"topic_{idx}"),
                "query": item.get("query", ""),
                "docs": docs_ordered,
                "doc_set": set(docs_ordered),
                "summary": item.get("summary", ""),
            }
        )
    return nodes


def compute_corpus_idf(nodes: List[Dict[str, Any]], eps: float = 1e-9) -> Dict[str, float]:
    df: Dict[str, int] = {}
    for n in nodes:
        for d in n["doc_set"]:
            df[d] = df.get(d, 0) + 1
    V = len(nodes)
    idf = {d: math.log((V + 1) / (cnt + 1)) for d, cnt in df.items()}
    return {d: (v if v > eps else eps) for d, v in idf.items()}


def build_tfidf_edges(
    nodes: List[Dict[str, Any]],
    idf: Dict[str, float],
    min_weight: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tf_list: List[Dict[str, float]] = []
    denom: List[float] = []
    for A in nodes:
        tf_A = {}
        for r, d in enumerate(A["docs"], start=1):
            tf_A[d] = rank_discount(r)
        tf_list.append(tf_A)
        RA = A["doc_set"]
        denom_A = sum(tf_A.get(d, 0.0) * idf.get(d, 0.0) for d in RA)
        denom.append(denom_A if denom_A > 0 else 0.0)

    edges_undirected: List[Dict[str, Any]] = []
    edges_directed: List[Dict[str, Any]] = []

    n = len(nodes)
    for i in range(n):
        A = nodes[i]
        RA = A["doc_set"]
        if not RA or denom[i] <= 0:
            continue
        for j in range(i + 1, n):
            B = nodes[j]
            RB = B["doc_set"]
            if not RB or denom[j] <= 0:
                continue

            inter = RA.intersection(RB)

            supp_A = sum(tf_list[i].get(d, 0.0) * idf.get(d, 0.0) for d in inter)
            supp_B = sum(tf_list[j].get(d, 0.0) * idf.get(d, 0.0) for d in inter)

            w_ab = supp_A / denom[i] if denom[i] > 0 else 0.0
            w_ba = supp_B / denom[j] if denom[j] > 0 else 0.0

            if w_ab >= min_weight:
                edges_directed.append(
                    {
                        "src_id": A["id"],
                        "tgt_id": B["id"],
                        "weight": round(w_ab, 6),
                    }
                )
            if w_ba >= min_weight:
                edges_directed.append(
                    {
                        "src_id": B["id"],
                        "tgt_id": A["id"],
                        "weight": round(w_ba, 6),
                    }
                )

            w = 0.5 * (w_ab + w_ba)
            if w < min_weight:
                continue
            edges_undirected.append(
                {
                    "nodeA": A["id"],
                    "nodeB": B["id"],
                    "weight": round(w, 6),
                    "overlap_count": len(inter),
                    "docs_A": len(RA),
                    "docs_B": len(RB),
                }
            )

    return edges_undirected, edges_directed

def load_concept_adj(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = {}
    for k, v in data.items():
        if not isinstance(v, list):
            continue
        cleaned[str(k)] = [str(x) for x in v]
    return cleaned


def build_concept_topic_edges(concept_adj: Dict[str, List[str]]) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    for src, dst_list in concept_adj.items():
        for dst in dst_list:
            if not src or not dst :
                continue
            edges.add(edge_key(src, dst))
    return edges


def build_concept_directed_edges_on_ids(
    nodes: List[Dict[str, Any]],
    concept_adj: Dict[str, List[str]],
    concept_weight: float = 0.02,
) -> List[Dict[str, Any]]:
    topic2ids: Dict[str, List[str]] = {}
    for n in nodes:
        t = n.get("topic")
        nid = n.get("id")
        if not t or not nid:
            continue
        topic2ids.setdefault(t, []).append(nid)

    edges: List[Dict[str, Any]] = []
    for src_topic, dst_list in concept_adj.items():
        src_ids = topic2ids.get(src_topic)
        if not src_ids:
            continue
        for dst_topic in dst_list:
            dst_ids = topic2ids.get(dst_topic)
            if not dst_ids:
                continue
            for sid in src_ids:
                for tid in dst_ids:
                    if sid == tid:
                        continue
                    edges.append(
                        {
                            "src_id": sid,
                            "tgt_id": tid,
                            "weight": float(concept_weight),
                        }
                    )
    print(f"[INFO] Concept directed edges (id-level) = {len(edges)}")
    return edges

def compute_pagerank_directed(
    nodes: List[Dict[str, Any]],
    directed_edges: List[Dict[str, Any]],
    alpha: float = 0.85,
) -> Dict[str, float]:
    G = nx.DiGraph()
    for n in nodes:
        nid = n.get("id")
        if nid:
            G.add_node(nid)
    for e in directed_edges:
        u = e.get("src_id")
        v = e.get("tgt_id")
        if not u or not v or u == v:
            continue
        w = float(e.get("weight", 0.0))
        if w <= 0:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    if G.number_of_nodes() == 0:
        print("[WARN] Directed graph has no nodes, PageRank returns empty.")
        return {}

    print(f"[INFO] Running PageRank on directed graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    pr = nx.pagerank(G, alpha=alpha, weight="weight")
    if pr:
        vals = list(pr.values())
        print(
            f"[INFO] PageRank stats: "
            f"min={min(vals):.6f}, max={max(vals):.6f}, mean={sum(vals)/len(vals):.6f}"
        )
    return pr

def merge_tfidf_and_concept(
    nodes: List[Dict[str, Any]],
    tfidf_edges_undirected: List[Dict[str, Any]],
    concept_topic_edges: Set[Tuple[str, str]],
    concept_min_weight: float,
    add_concept_only: bool,
):
    id2topic: Dict[str, str] = {}
    topic2ids: Dict[str, List[str]] = {}
    for n in nodes:
        nid = n.get("id")
        t = n.get("topic")
        if not nid or not t:
            continue
        id2topic[nid] = t
        topic2ids.setdefault(t, []).append(nid)

    tfidf_edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in tfidf_edges_undirected:
        idA = e.get("nodeA")
        idB = e.get("nodeB")
        if not idA or not idB or idA == idB:
            continue
        tA = id2topic.get(idA)
        tB = id2topic.get(idB)
        if not tA or not tB:
            continue
        key = edge_key(tA, tB)
        w = float(e.get("weight", 0.0))
        if key not in tfidf_edge_map or w > float(tfidf_edge_map[key].get("weight", 0.0)):
            tfidf_edge_map[key] = e

    merged_edges: List[Dict[str, Any]] = []
    for key, edge in tfidf_edge_map.items():
        w_tfidf = float(edge.get("weight", 0.0))
        if key in concept_topic_edges and w_tfidf < concept_min_weight:
            new_w = concept_min_weight
        else:
            new_w = w_tfidf
        new_edge = dict(edge)
        new_edge["weight"] = round(new_w, 6)
        merged_edges.append(new_edge)

    if add_concept_only:
        existing = set(tfidf_edge_map.keys())
        for key in concept_topic_edges:
            if key in existing:
                continue
            tA, tB = key
            idsA = topic2ids.get(tA)
            idsB = topic2ids.get(tB)
            if not idsA or not idsB:
                continue
            merged_edges.append(
                {
                    "nodeA": idsA[0],
                    "nodeB": idsB[0],
                    "weight": round(concept_min_weight, 6),
                }
            )

    nodes_out = []
    for n in nodes:
        nodes_out.append(
            {
                "id": n.get("id"),
                "topic": n.get("topic"),
                "query": n.get("query"),
                "doc_count": len(n.get("docs", [])),
                "summary": n.get("summary"),
            }
        )

    return nodes_out, merged_edges

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--embedding_prefix", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--stance", type=str, default="oppose", choices=["support", "oppose"])
    parser.add_argument("--concept_min_weight", type=float, default=0.20)
    parser.add_argument("--concept_weight", type=float, default=0.08)
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--idf_eps", type=float, default=1e-9)
    parser.add_argument("--add_concept_only", action="store_true", default=True) 
    
    args = parser.parse_args()

    topic_dir = os.path.join(args.base_dir, args.category, args.topic)
    results_path = os.path.join(topic_dir, f"{args.topic}_top50_results.json")
    concept_path = os.path.join(topic_dir, f"kg_adj_{args.topic}.json")
    
    clean_filename = f"Llama-31_{args.topic}_top5_clean.json"
    clean_path = os.path.join(topic_dir, clean_filename)

    output_path = os.path.join(topic_dir, f"graph_{args.stance}.json")

    print(f"========== Pipeline Step 1: Graph Build (Filtered) ==========")
    print(f"Topic: {args.topic}")
    print(f"Stance Target: {args.stance}")
    print(f"Raw Input: {results_path}")
    print(f"Clean Eval: {clean_path}")
    print(f"Concept Input: {concept_path}")
    print(f"Output Graph: {output_path}")

    if not os.path.exists(results_path):
        print(f"[Error] Results file not found -> {results_path}")
        return
    if not os.path.exists(concept_path):
        print(f"[Error] Concept file not found -> {concept_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    filtered_raw = filter_raw_data_by_stance(raw, clean_path, args.stance)
    
    nodes = build_nodes(filtered_raw)

    if not nodes:
        print("[Error] No nodes left after filtering. Exiting.")
        return

    idf = compute_corpus_idf(nodes, eps=args.idf_eps)
    tfidf_edges_undirected, tfidf_edges_directed = build_tfidf_edges(nodes, idf, min_weight=args.min_weight)
    print(f"[INFO] TF-IDF built: nodes={len(nodes)}")

    concept_adj = load_concept_adj(concept_path)
    concept_topic_edges = build_concept_topic_edges(concept_adj)
    concept_dir_edges = build_concept_directed_edges_on_ids(nodes, concept_adj, concept_weight=args.concept_weight)

    nodes_out, merged_edges = merge_tfidf_and_concept(
        nodes,
        tfidf_edges_undirected,
        concept_topic_edges,
        concept_min_weight=args.concept_min_weight,
        add_concept_only=args.add_concept_only,
    )

    all_directed_edges = tfidf_edges_directed + concept_dir_edges
    pr_scores = compute_pagerank_directed(nodes, all_directed_edges, alpha=0.85)

    if pr_scores:
        max_pr = max(pr_scores.values())
        pr_norm = {nid: (score / max_pr) for nid, score in pr_scores.items()} if max_pr > 0 else {}
    else:
        pr_norm = {}

    for n in nodes_out:
        nid = n.get("id")
        n["weight"] = round(float(pr_norm.get(nid, 0.0)), 6)

    merged = {"nodes": nodes_out, "edges": merged_edges}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Filtered Graph built successfully: {output_path}")

if __name__ == "__main__":
    main()