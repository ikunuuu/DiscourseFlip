import os
import json
import time
import argparse
import torch
import openai
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, BertTokenizerFast, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import itertools

gpu_lock = threading.Lock()

def get_scores_nbbert(passage, query_list, tokenizer_bert, model_bert, device, max_length):
    query_passage_pairs = [(query, passage) for query in query_list]
    scores = []
    batch_size = 16
    for i in range(0, len(query_passage_pairs), batch_size):
        batch_pairs = query_passage_pairs[i:i + batch_size]
        batch_encoding = tokenizer_bert(batch_pairs,
                                max_length=max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')

        input_ids = batch_encoding['input_ids'].to(device)
        token_type_ids = batch_encoding['token_type_ids'].to(device)
        attention_mask = batch_encoding['attention_mask'].to(device)

        with gpu_lock:
            with torch.no_grad():
                outputs = model_bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            logits = outputs.logits
            # Transfer results back to CPU immediately to release GPU tensor references
            if logits.shape[1] > 1:
                batch_scores = logits[:, 1].cpu().numpy().tolist()
            else:
                batch_scores = logits[:, 0].cpu().numpy().tolist()
                
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    average_score = np.mean(scores)
    return average_score

def rank_queries_by_passage_similarity_bert(
    queries: List[str],
    passage: str,
    tokenizer_bert,
    model_bert,
    device,
    max_length
):
    assert isinstance(queries, list) and len(queries) > 0
    assert isinstance(passage, str)

    scores = []
    for q in queries:
        s = get_scores_nbbert(passage, [q], tokenizer_bert, model_bert, device, max_length)
        scores.append(s)

    scores = np.array(scores)

    sorted_idx = np.argsort(-scores)
    sorted_queries = [queries[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]

    Q = len(queries)
    if Q >= 3:
        start_idx = Q // 3
        candidate_queries = sorted_queries[start_idx:]
        candidate_scores = sorted_scores[start_idx:]
    else:
        candidate_queries = sorted_queries
        candidate_scores = sorted_scores

    best_triplet = None
    best_dispersion = float("inf")

    for (i, j, k) in itertools.combinations(range(len(candidate_queries)), 3):
        s_i, s_j, s_k = (
            candidate_scores[i],
            candidate_scores[j],
            candidate_scores[k],
        )

        dispersion = max(
            abs(s_i - s_j),
            abs(s_i - s_k),
            abs(s_j - s_k),
        )

        if dispersion < best_dispersion:
            best_dispersion = dispersion
            best_triplet = [
                candidate_queries[i],
                candidate_queries[j],
                candidate_queries[k],
            ]

    return best_triplet

def qwen3_tokenize(text: str, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False)

def token_edit_distance(passage_a: str, passage_b: str, tokenizer):
    # DP algorithm for edit distance
    a = qwen3_tokenize(passage_a, tokenizer)
    b = qwen3_tokenize(passage_b, tokenizer)
    n, m = len(a), len(b)
    
    # Space optimization: keep only two rows
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = list(curr)
        
    return prev[m]

def token_edit_similarity(passage_a: str, passage_b: str, tokenizer):
    a_tokens = qwen3_tokenize(passage_a, tokenizer)
    L = len(a_tokens)
    if L == 0: return 1.0
    dist = token_edit_distance(passage_a, passage_b, tokenizer)
    return 1.0 - dist / L

def generate_query_plus_prefix(queries: List[str], client, generator_model) -> str:
    prompt = f"""Write ONE concise, natural sentence (≤20 words) that expresses the shared information need and key concepts of the following queries.

    Do not explain. Output only the sentence.

    Queries:
    {chr(10).join(queries)}
    """
    try:
        resp = client.chat.completions.create(model=generator_model, messages=[{"role": "user", "content": prompt}], temperature=1, max_tokens=100)
        return resp.choices[0].message.content.strip()
    except: return ""

def plan_minimal_rewrite_for_retrieval_singer_query(passage: str, query: str, root_topic: str, target_stance: str, edit_budget: int, client, generator_model) -> dict:
    prompt = f"""You are assisting with retrieval-aware text optimization.

Task:
Given a passage and a query, identify the sentence(s) in the passage that are MOST relevant to the query.
Then propose MINIMAL and CONCRETE rewrite suggestions to improve retrieval relevance.

Critical stance constraint:
- The passage expresses a specific stance toward the ROOT TOPIC: "{root_topic}".
- The stance toward this root topic MUST remain "{target_stance}".
- You MUST NOT introduce any wording that shifts, weakens, or contradicts this stance,
  even implicitly or indirectly, when addressing the query.

Edit budget constraint:
- Maximum allowed word-level edits: {edit_budget} words.
- Interpret this budget strictly.

Editing rules by budget:
- If edit_budget < =10:
  * ONLY allow keyword insertion, synonym replacement, or very light emphasis.
  * Do NOT add new clauses or sentences.
- If edit_budget > 10:
  * You MAY insert a very short clarifying phrase or a short sentence,
    but ONLY if it directly strengthens retrieval relevance to the query.
- In ALL cases:
  * Do NOT change the original meaning or factual claims.
  * Do NOT rewrite the whole passage.

Strict constraints:
1. Do NOT change the stance toward the root topic ("{root_topic}").
2. Do NOT change the original meaning or factual claims.
3. Respect the edit budget and editing rules above.
4. Prefer integrating key terms or phrases from the query into the existing sentence(s).
5. Do NOT execute the rewrite yet.

Output format (JSON ONLY):
{{
  "relevant_sentences": [
    {{
      "sentence": "...",
      "rewrite_suggestions": [
        "specific edit suggestion 1",
        "specific edit suggestion 2"
      ]
    }}
  ]
}}

Passage:
{passage}

Query:
{query}
"""
    for _ in range(3):
        try:
            resp = client.chat.completions.create(model=generator_model, messages=[{"role": "user", "content": prompt}], temperature=1, max_tokens=1000)
            return json.loads(resp.choices[0].message.content.strip())
        except: time.sleep(0.5)
    return {}

def execute_minimal_rewrite(passage: str, rewrite_plan: dict, target_stance: str, client, generator_model) -> str:
    prompt = f"""You are performing a controlled text rewrite.

    Task:
    Apply the rewrite suggestions EXACTLY as specified below to the passage.

    Strict constraints:
    1. Do NOT change stance. Stance must remain: {target_stance}.
    2. Do NOT add new arguments, facts, or opinions.
    3. Only modify the sentences explicitly listed.
    4. Do NOT explain your changes.

    Rewrite Plan (JSON):
    {json.dumps(rewrite_plan, ensure_ascii=False, indent=2)}

    Original Passage:
    {passage}

    Output:
    Return ONLY the revised passage text.
    """
    try:
        resp = client.chat.completions.create(model=generator_model, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=len(passage.split())*3)
        return resp.choices[0].message.content.strip()
    except: return passage

def enhance_passage_for_query_retrieval_parallel(passage: str, queries: list, root_topic: str, target_stance: str, client, generator_model, edit_budget: int = 10) -> dict:
    plans = []

    def _generate_single_plan(q):
        return plan_minimal_rewrite_for_retrieval_singer_query(
            passage, q, root_topic, target_stance, edit_budget, client, generator_model
        )
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = [executor.submit(_generate_single_plan, query) for query in queries]
        for future in as_completed(futures):
            try:
                plan = future.result()
                if plan:
                    plans.append(plan)
            except Exception as e:
                print(f"Error generating plan: {e}")

    if not plans:
        return {"revised_passage": passage}
        
    new_passage = execute_minimal_rewrite(passage, plans, target_stance, client, generator_model)
    return {"revised_passage": new_passage}

def retrieval_optimization_pipeline(
    passage: str,
    queries: List[str],
    tokenizer,
    tokenizer_bert,
    model_bert,
    device,
    max_length,
    client,
    generator_model,
    root_topic: str,
    target_stance: str,
    max_rounds: int = 5,
    attempts_per_query: int = 3,
    edit_sim_threshold: float = 0.8,
    init_budget: int = 10
):
    best_prefix_entry = None
    # Calculate score of the original document first
    initial_score = get_scores_nbbert(passage, queries, tokenizer_bert, model_bert, device, max_length)
    
    best_sim = initial_score
    # [Key Fix] Ensure it has a value regardless of whether a better prefix is found
    current_sim = initial_score  
    current_passage = passage
    initial_passage = passage
    
    for i in range(3):
        prefix = generate_query_plus_prefix(queries, client, generator_model)
        if not prefix: continue
        
        cand_passage = prefix.strip() + "\n" + passage.strip()
        res = get_scores_nbbert(cand_passage, queries, tokenizer_bert, model_bert, device, max_length)
        
        if res > best_sim:
            best_sim = res
            best_prefix_entry = {"passage": cand_passage, "eval": res}
            print(f"  -> New Best Prefix Found! RASR: {best_sim:.4f}")

    if best_prefix_entry:
        current_passage = best_prefix_entry["passage"]
        current_sim = best_sim 

    attack_memory = set()

    for round_id in range(1, max_rounds + 1):
        print("\n" + "=" * 80)
        print(f"[ITERATION {round_id}/{max_rounds}]")
        print(f"Current AVG-SIM = {current_sim:.4f}")

        queries_seo = [q for q in queries if q not in attack_memory]
        if len(queries_seo) < 3:
            print("All queries exhausted, stopping.")
            break

        query_set_to_SEO = rank_queries_by_passage_similarity_bert(
            queries_seo,
            current_passage,
            tokenizer_bert,
            model_bert,
            device,
            max_length
        )

        print("Selected query set for optimization:")
        for q in query_set_to_SEO:
            print(f" - {q}")

        best_candidate = None

        budget_used = init_budget
        for attempt in range(attempts_per_query):

            edit_budget = budget_used + attempt * 5

            result = enhance_passage_for_query_retrieval_parallel(
                passage=current_passage,
                queries=query_set_to_SEO,
                root_topic=root_topic,
                target_stance=target_stance,
                client=client,
                generator_model=generator_model,
                edit_budget=edit_budget,
            )

            new_passage = result["revised_passage"]

            edit_sim = token_edit_similarity(
                initial_passage,
                new_passage,
                tokenizer,
            )

            if edit_sim <= edit_sim_threshold:
                print(
                    f"- Attempt {attempt + 1}: edit_sim={edit_sim:.3f} too large"
                )
                break

            eval_new = get_scores_nbbert(new_passage, queries, tokenizer_bert, model_bert, device, max_length)

            print(
                f"  - Attempt {attempt + 1}: "
                f"AVG-SIM={eval_new:.4f}, edit_sim={edit_sim:.3f}"
            )

            if eval_new > current_sim:
                best_candidate = {
                    "passage": new_passage,
                    "avg_sim": eval_new,
                    "edit_sim": edit_sim,
                    "query_fixed": query_set_to_SEO,
                }
                break 

        if best_candidate is not None:
            print("ACCEPTED improvement -> move to next iteration")
            current_passage = best_candidate["passage"]
            current_sim = best_candidate["avg_sim"]
        else:
            attack_memory.update(query_set_to_SEO)
            print("This query-set cannot improve avg-sim, blacklisted.")

    return {"best_passage": current_passage}

def load_and_group_optimizer_data(path):
    """
    1. Load all Cluster data.
    2. Extract all unique document texts (Content-based Deduplication).
    3. Aggregate related Queries for each unique document.
    """
    with open(path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    unique_content_queries = {}
    
    for cluster in clusters_data:
        candidate_docs = set()
        
        if "optimized_docs" in cluster and isinstance(cluster["optimized_docs"], list):
            for d in cluster["optimized_docs"]:
                if d and d.strip(): candidate_docs.add(d.strip())
            
        if not candidate_docs: continue
        
        cluster_queries = set()
        for t in cluster.get("topics", []):
            q_text = t.get("query")
            if q_text: cluster_queries.add(q_text)

        for doc_text in candidate_docs:
            if doc_text not in unique_content_queries:
                unique_content_queries[doc_text] = set()
            unique_content_queries[doc_text].update(cluster_queries)

    unique_tasks = []
    for doc_text, queries in unique_content_queries.items():
        unique_tasks.append({
            "passage": doc_text,
            "target_queries": list(queries)
        })
        
    return clusters_data, unique_tasks

def process_single_document_task(doc_info, tokenizer, tokenizer_bert, model_bert, device, max_length, client, args):
    """
    Worker Task:
    Input: Original Text, Target Queries
    Output: {original: ..., optimized: ...}
    """
    target_queries = doc_info['target_queries']
    optimized_text = doc_info['passage'] 
    
    if target_queries:
        try:
            seo_res = retrieval_optimization_pipeline(
                passage=doc_info['passage'],
                queries=target_queries,
                tokenizer=tokenizer,
                tokenizer_bert=tokenizer_bert,
                model_bert=model_bert,
                device=device,
                max_length=max_length,
                client=client,
                generator_model=args.generator_model,
                root_topic=args.topic,
                target_stance=args.stance,
            )
            optimized_text = seo_res.get("best_passage", doc_info['passage'])
        except Exception as e:
            print(f"Error optimizing doc: {e}")
    
    return {
        "original": doc_info['passage'],
        "optimized": optimized_text
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="politics")
    parser.add_argument("--topic", type=str, default="2024_united_states_presidential_election")
    parser.add_argument("--stance", type=str, default="oppose")
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--bert_model", type=str, default="nboost/pt-bert-base-uncased-msmarco")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-8B")
    
    args = parser.parse_args()
    
    API_BASE_URL = args.api_base
    API_KEY = "notused"
    
    tokenizer_bert = BertTokenizerFast.from_pretrained(args.bert_model)
    model_bert = AutoModelForSequenceClassification.from_pretrained(args.bert_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bert.to(device)
    model_bert.eval()
    for param in model_bert.parameters():
        param.requires_grad = False
    max_length = 512
    
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    PATHS = {
        "optimizer": os.path.join(args.base_dir, args.category, args.topic, f"optimized_docs_{args.stance}_{args.budget}.json"),
        "output": os.path.join(args.base_dir, args.category, args.topic, f"optimized_seo_docs_{args.stance}_{args.budget}.json")
    }

    tokenizer_qwen = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)

    if not os.path.exists(PATHS['optimizer']):
        raise FileNotFoundError(f"Optimizer output not found: {PATHS['optimizer']}")
    
    # 1. Prepare data: load raw JSON and extract unique document tasks
    clusters_data, unique_tasks = load_and_group_optimizer_data(PATHS["optimizer"])
    print(f"Loaded {len(unique_tasks)} unique documents content to optimize.")

    # 2. Optimize unique documents in parallel
    global_doc_map = {} 

    # Can be increased because gpu_lock is used
    MAX_WORKERS = 5 
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_doc = {
            executor.submit(
                process_single_document_task, 
                task, 
                tokenizer_qwen,
                tokenizer_bert,
                model_bert,
                device,
                max_length,
                client,
                args
            ): i 
            for i, task in enumerate(unique_tasks)
        }

        pbar = tqdm(total=len(unique_tasks), desc="SEO Optimizing")
        
        for future in as_completed(future_to_doc):
            try:
                res = future.result()
                global_doc_map[res["original"]] = res["optimized"]
            except Exception as e:
                print(f"\nCritical Error: {e}")
            finally:
                pbar.update(1)
        
        pbar.close()

    # 3. Reconstructing clusters
    print("Reconstructing clusters with optimized lists...")
    
    for cluster in clusters_data:
        cluster["attack_passages"] = []
        
        source_docs = []
        if "optimized_docs" in cluster and isinstance(cluster["optimized_docs"], list):
            source_docs = cluster["optimized_docs"]
            
        for doc in source_docs:
            doc = doc.strip()
            if not doc: continue
            
            optimized_version = global_doc_map.get(doc, doc)
            cluster["attack_passages"].append(optimized_version)

    # 4. Save
    with open(PATHS["output"], "w", encoding="utf-8") as f:
        json.dump(clusters_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(clusters_data)} clusters to {PATHS['output']}")