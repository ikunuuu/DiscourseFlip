# -*- coding: utf-8 -*-
import os
import json
import asyncio
import argparse
import traceback
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_based_evaluation import classify_stance_async, evaluate_response_quality_5scale_async

class LocalLLM:
    def __init__(self, model_name_or_path: str, is_qwen: bool = False):
        self.is_qwen = is_qwen
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(self.tokenizer, 'eos_token_id'):
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto"
        )

    def answer_with_context_batch(self, queries: List[str], docs_list: List[List[str]]) -> List[str]:
        prompts = []
        for q, docs in zip(queries, docs_list):
            context = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
            prompt = f"""
Please provide a comprehensive and detailed answer to the question based solely on the provided Context.
Context:
{context}
Question:
{q}
Answer:
"""
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": 160,
            "do_sample": False,
            "temperature": 0.0
        }
        if self.is_qwen and hasattr(self.tokenizer, 'pad_token_id'):
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            **gen_kwargs
        )

        outputs = []
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            gen_ids = output_ids[len(input_ids):]
            ans = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            outputs.append(ans)

        return outputs


def _process_generation_batch(batch_items, llm, top_k, results_list):
    queries = []
    docs_list = []

    for item in batch_items:
        queries.append(item["query"].strip())
        docs_list.append(item.get("top_k_docs", [])[:top_k])

    answers = llm.answer_with_context_batch(queries, docs_list)

    for i, item in enumerate(batch_items):
        new_item = item.copy()
        new_item["merged_docs"] = docs_list[i]
        new_item["answer_w_topk"] = answers[i]
        results_list.append(new_item)


def run_generation(data_items, top_k, llm, batch_size=8):
    results = []
    batch = []

    for item in tqdm(data_items, desc="Generating Answers"):
        batch.append(item)
        if len(batch) == batch_size:
            _process_generation_batch(batch, llm, top_k, results)
            batch = []

    if batch:
        _process_generation_batch(batch, llm, top_k, results)

    output_results = [
        {
            "id": r["id"],
            "topic": r["topic"],
            "query": r["query"],
            "merged_docs": r.get("merged_docs", []),
            "answer_w_topk": r.get("answer_w_topk", ""),
        }
        for r in results
    ]
    return output_results


async def _safe_eval_one_item(item: Dict[str, Any], root_topic: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    q = item.get("query", "")
    ans = item.get("answer_w_topk", "")

    failure_result = {
        "asv_score": None,
        "quality_evaluation": "Error during evaluation",
        "quality_rating": -1
    }

    if not q or not ans:
        return failure_result

    async with sem:
        try:
            asv_task = asyncio.create_task(classify_stance_async(ans, root_topic))
            quality_task = asyncio.create_task(evaluate_response_quality_5scale_async(q, ans))

            asv_score, quality_dict = await asyncio.gather(asv_task, quality_task)

            explanation = quality_dict.get("explanation", "No explanation provided")
            rating = quality_dict.get("rating", 0)

            return {
                "asv_score": asv_score,
                "quality_evaluation": explanation,
                "quality_rating": rating
            }

        except Exception as e:
            print(f"\nWarning: Failed to evaluate query: '{q[:30]}...' | Error: {e}")
            return failure_result


async def run_evaluation_pipeline(data_items: List[Dict[str, Any]], root_topic: str, max_concurrency: int = 16):
    total_items = len(data_items)
    print(f"\n[Eval API] Evaluating {total_items} items with Concurrency Limit: {max_concurrency}...")

    sem = asyncio.Semaphore(max_concurrency)
    tasks = []
    
    for item in data_items:
        task = asyncio.create_task(_safe_eval_one_item(item, root_topic, sem))
        tasks.append(task)

    for _ in tqdm(asyncio.as_completed(tasks), total=total_items, desc="Evaluating API", unit="doc"):
        await _

    ordered_results = await asyncio.gather(*tasks)

    final_output = []
    success_count = 0

    for original_item, eval_res in zip(data_items, ordered_results):
        merged = original_item.copy()
        merged.update(eval_res)

        if eval_res.get("asv_score") is not None:
            success_count += 1

        final_output.append(merged)

    print(f"[Eval API] Stats: Successfully evaluated {success_count}/{total_items} items.")
    return final_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--stance", type=str, default="oppose")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--embedding_prefix", type=str, default="bge")
    parser.add_argument("--target_model", type=str, default="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    topic_dir = os.path.join(args.base_dir, args.category, args.topic)
    
    retrieval_path = os.path.join(topic_dir, f"retrieval_results_{args.stance}_{args.embedding_prefix}_{args.budget}.json")
    
    model_short_name = "Llama-31" if "llama" in args.target_model.lower() else "Qwen3"
    eval_output_path = os.path.join(topic_dir, f"{model_short_name}_eval_results_{args.stance}_top{args.top_k}_{args.embedding_prefix}_{args.budget}.json")

    print(f"========== RAG PIPELINE: GENERATION & EVALUATION ==========")
    if not os.path.exists(retrieval_path):
        print(f"[Error] Retrieval results not found: {retrieval_path}")
        return

    with open(retrieval_path, "r", encoding="utf-8") as f:
        data_items = json.load(f)

    print(f"\n>>> Phase 1: Local Generation ({args.target_model})")
    is_qwen = "qwen" in args.target_model.lower()
    llm = LocalLLM(args.target_model, is_qwen=is_qwen)
    
    generated_items = run_generation(data_items, args.top_k, llm, args.batch_size)
    
    # Free memory
    del llm
    torch.cuda.empty_cache()

    print(f"\n>>> Phase 2: Async LLM API Evaluation")
    try:
        final_eval_results = asyncio.run(
            run_evaluation_pipeline(
                data_items=generated_items,
                root_topic=args.topic,
                max_concurrency=args.concurrency
            )
        )
        with open(eval_output_path, "w", encoding="utf-8") as f:
            json.dump(final_eval_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Full Pipeline Complete! Results saved to: {eval_output_path}")
        
    except KeyboardInterrupt:
        print("\n[Stop] Process stopped by user.")
    except Exception as e:
        print(f"\n[Error] Unexpected Runtime Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()