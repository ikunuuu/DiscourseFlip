# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

RETURN_TOPK = 50

def clean_content(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return "\n".join(lines).strip()

def load_ourrag_poison_data(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    poisoned_chunks = []
    seen_passages = set()
    for idx, item in enumerate(data_list):
        for raw_passage in item.get("attack_passages", []):
            cleaned_doc = clean_content(raw_passage)
            if not cleaned_doc:
                continue
            if cleaned_doc in seen_passages:
                continue
            seen_passages.add(cleaned_doc)
            poisoned_chunks.append({
                "id": f"poison_{idx}",
                "url": "POISONED_SOURCE",
                "title": f"Generated_Doc_{idx}",
                "content": cleaned_doc,
                "content_chunk": [cleaned_doc]
            })
    return poisoned_chunks

class UniversalEmbeddingWrapper(Embeddings):
    def __init__(self, model_path: str, device: str = None, batch_size: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_name_lower = model_path.lower()
        self.batch_size = batch_size
        if "contriever" in self.model_name_lower:
            self.engine_type = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
        else:
            self.engine_type = "sentence_transformers"
            self.model = SentenceTransformer(model_path, device=self.device, trust_remote_code=True)
            self.encode_kwargs = {"normalize_embeddings": True, "batch_size": self.batch_size}

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

    def _embed_contriever(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            emb = self._mean_pooling(out, enc["attention_mask"])
            all_embeddings.extend(emb.tolist())
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.engine_type == "transformers":
            return self._embed_contriever(texts)
        return self.model.encode(texts, **self.encode_kwargs).tolist()

    def embed_query(self, text: str) -> List[float]:
        if self.engine_type == "transformers":
            return self._embed_contriever([text])[0]
        return self.model.encode([text], **self.encode_kwargs)[0].tolist()

def inject_poison_into_existing_index(clean_index_path, poisoned_chunks, model_path):
    embedder = UniversalEmbeddingWrapper(model_path=model_path)
    vectorstore = FAISS.load_local(
        clean_index_path,
        embedder,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    texts, metas = [], []
    for rec in poisoned_chunks:
        for chunk in rec["content_chunk"]:
            texts.append(chunk)
            metas.append({
                "source_url": "POISONED_SOURCE",
                "doc_id": rec["id"]
            })
    if texts:
        vectorstore.add_texts(texts, metadatas=metas)
    return vectorstore

def retrieve_topk(nodes, vectorstore, poisoned_chunks, k=50):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    outputs = []
    import random
    random.seed(42)
    for node in tqdm(nodes, desc="Retrieving Documents"):
        queries = node.get("queries") or []
        if not queries:
            continue
        query = random.choice(queries)
        docs = retriever.invoke(query)
        hits = [d.page_content.strip() for d in docs]
        outputs.append({
            "id": node.get("id"),
            "topic": node.get("wiki_title"),
            "query": query,
            "top_k_docs": hits,
            "summary": node.get("summary"),
        })
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--stance", type=str, default="oppose")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--embedding_prefix", type=str, default="bge")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    topic_dir = os.path.join(args.base_dir, args.category, args.topic)
    clean_index_path = os.path.join(topic_dir, "faiss_index")
    topic_tree_path = os.path.join(topic_dir, f"topic_tree_{args.topic}_with_summary_queries.json")
    poison_file = os.path.join(topic_dir, f"optimized_docs_{args.stance}_{args.budget}.json")
    
    output_path = os.path.join(topic_dir, f"retrieval_results_{args.stance}_{args.embedding_prefix}_{args.budget}.json")

    print(f"[Retrieval] Loading nodes from {topic_tree_path}")
    with open(topic_tree_path, "r", encoding="utf-8") as f:
        query_nodes = json.load(f)

    print(f"[Retrieval] Loading poisoned documents from {poison_file}")
    poisoned_chunks = load_ourrag_poison_data(poison_file)
    
    print(f"[Retrieval] Injecting poisons into FAISS index...")
    vectorstore = inject_poison_into_existing_index(
        clean_index_path,
        poisoned_chunks,
        args.embed_model
    )
    
    results = retrieve_topk(query_nodes, vectorstore, poisoned_chunks, RETURN_TOPK)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Retrieval] Completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()