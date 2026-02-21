# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))
from llm_based_evaluation import classify_stance_async, evaluate_response_quality_5scale_async

class LocalLLM:
    def __init__(self, model_name_or_path: str, is_qwen: bool = False):
        print(f"[Clean RAG] Loading LLM from {model_name_or_path}...")
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
        print("[Clean RAG] ✅ LLM Loaded.")

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


def generate_clean_answers(data_items: List[Dict], llm: LocalLLM, top_k: int, batch_size: int):
    results = []
    batch_items = []
    
    def _process_batch(batch):
        queries = [item["query"].strip() for item in batch]
        docs_list = [item.get("top_k_docs", [])[:top_k] for item in batch]
        answers = llm.answer_with_context_batch(queries, docs_list)
        
        for i, item in enumerate(batch):
            new_item = item.copy()
            new_item["merged_docs"] = docs_list[i]  # For Clean RAG, merged_docs = top_k_docs
            new_item["answer_w_topk"] = answers[i]
            results.append(new_item)

    for item in tqdm(data_items, desc="🚀 Generating Clean Answers"):
        batch_items.append(item)
        if len(batch_items) == batch_size:
            _process_batch(batch_items)
            batch_items = []
            
    if batch_items:
        _process_batch(batch_items)
        
    return results

async def _eval_one_item_async(item: Dict[str, Any], root_topic: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    q = item.get("query")
    ans = item.get("answer_w_topk")
    
    if not q or not ans:
        return item

    async with sem:
        try:
            asv_task = asyncio.create_task(classify_stance_async(ans, root_topic))
            quality_task = asyncio.create_task(evaluate_response_quality_5scale_async(q, ans))
            asv_score, quality_dict = await asyncio.gather(asv_task, quality_task)
            
            item["asv_score"] = asv_score
            item["quality_evaluation"] = quality_dict.get("explanation")
            item["quality_rating"] = quality_dict.get("rating")
        except Exception as e:
            print(f"⚠️ Error evaluating query: {q[:20]}... {e}")
            item["error"] = "eval_failed"
            
    return item

async def run_evaluation(data_items: List[Dict], root_topic: str, concurrency: int):
    print(f"🧪 Evaluating {len(data_items)} items via API (Concurrency: {concurrency})...")
    sem = asyncio.Semaphore(concurrency)
    
    tasks = []
    for item in data_items:
        tasks.append(asyncio.create_task(_eval_one_item_async(item, root_topic, sem)))
            
    for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="📊 Evaluating API"):
        await _
        
    results = await asyncio.gather(*tasks)
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Clean RAG Baseline before Graph Build")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--target_model", type=str, default="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    parser.add_argument("--embed_prefix", type=str, default="bge")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    topic_dir = os.path.join(args.base_dir, args.category, args.topic)
    
    top_k_path = os.path.join(topic_dir, f"{args.topic}_top50_results.json")
    model_short_name = "Llama-31" if "llama" in args.target_model.lower() else "Qwen3"
    
    output_fname = f"{model_short_name}_{args.topic}_top{args.top_k}_clean.json"
    output_path = os.path.join(topic_dir, output_fname)

    print(f"========== Prep Phase: Clean RAG Evaluation ==========")
    print(f"📥 Clean Input: {top_k_path}")
    print(f"💾 Eval Output: {output_path}")

    if not os.path.exists(top_k_path):
        print(f"❌ Clean retrieval file not found: {top_k_path}")
        return

    with open(top_k_path, 'r', encoding='utf-8') as f: 
        data_topk = json.load(f)

    # 1. Generation
    is_qwen = "qwen" in args.target_model.lower()
    llm = LocalLLM(args.target_model, is_qwen=is_qwen)
    generated_items = generate_clean_answers(data_topk, llm, args.top_k, args.batch_size)
    
    # Free GPU memory before starting heavy async tasks (optional but recommended)
    del llm
    torch.cuda.empty_cache()

    # 2. Evaluation
    try:
        final_results = asyncio.run(run_evaluation(generated_items, args.topic, args.concurrency))
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Clean RAG Evaluation Complete. Saved to: {output_path}")
        
    except KeyboardInterrupt:
        print("\n🛑 Process stopped by user.")
    except Exception as e:
        print(f"\n❌ Unexpected Runtime Error: {e}")

if __name__ == "__main__":
    main()