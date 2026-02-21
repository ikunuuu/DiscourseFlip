# -*- coding: utf-8 -*-
import os
import json
import asyncio
import time
import argparse
import hashlib
from urllib.parse import quote, urlparse, urlunparse, parse_qsl, urlencode
from typing import List, Dict, Any, Set
from datetime import datetime

import aiohttp
import requests
import torch
import numpy as np
from dateutil.tz import tzutc
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer 
from langdetect import detect
from transformers import AutoTokenizer, AutoModel

JINA_API_KEY = os.environ.get("JINA_API_KEY")

SEARCH_TOPK = 10
CONCURRENCY = 500  
CHUNK_SIZE = 1536
CHUNK_OVERLAP = 200
TOKEN_CHUNK = 512
TOKEN_OVERLAP = 64
SEARCH_TIMEOUT = 30
RETURN_TOPK = 50

class UniversalEmbeddingWrapper(Embeddings):
    def __init__(self, model_path: str, device: str = None, batch_size: int = 64):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_path = model_path
        self.model_name_lower = model_path.lower()
        self.batch_size = batch_size
        
        print(f"[Embedding] Loading Model: {model_path} on {self.device}...")

        if "contriever" in self.model_name_lower:
            self.engine_type = "transformers"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path).to(self.device)
                self.model.eval() 
            except Exception as e:
                raise ValueError(f"Failed to load Contriever using Transformers: {e}")
        else:
            self.engine_type = "sentence_transformers"
            
            try:
                self.model = SentenceTransformer(model_path, device=self.device, trust_remote_code=True)
            except Exception as e:
                raise ValueError(f"Failed to load model using SentenceTransformer: {e}")
            
            self.encode_kwargs = {'normalize_embeddings': True, 'batch_size': self.batch_size}

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

    def _embed_contriever(self, texts: List[str]) -> List[List[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
        
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                 
            all_embeddings.extend(embeddings.tolist())
            
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.engine_type == "transformers":
            return self._embed_contriever(texts)
        else:
            return self.model.encode(texts, **self.encode_kwargs).tolist()

    def embed_query(self, text: str) -> List[float]:
        if "bge" in self.model_name_lower:
            text = f"Represent this sentence for searching relevant passages: {text}"
        
        if self.engine_type == "transformers":
            return self._embed_contriever([text])[0]
        else:
            if "qwen" in self.model_name_lower:
                return self.model.encode([text], prompt_name="query", **self.encode_kwargs)[0].tolist()
            
            return self.model.encode([text], **self.encode_kwargs)[0].tolist()

def get_urls(content: str, topk: int = SEARCH_TOPK, token: str = JINA_API_KEY) -> List[Dict[str, Any]]:
    base = f"https://s.jina.ai/?q={quote(content)}&num={max(1, min(topk, 50))}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}" if token else "",
        "X-Respond-With": "no-content",
    }
    try:
        resp = requests.get(base, headers=headers, timeout=SEARCH_TIMEOUT)
        resp.raise_for_status()
        js = resp.json()
        return js.get("data", []) or []
    except Exception as e:
        print(f"[get_urls] ERROR '{content}': {e}")
        return []

def _extract_url_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    url = item.get("url") or item.get("link") or item.get("u") or item.get("source")
    title = item.get("title") or item.get("t") or item.get("name") or ""
    snippet = item.get("snippet") or item.get("s") or item.get("description") or ""
    return {"url": url, "title": title, "snippet": snippet}

async def get_url_content(session: aiohttp.ClientSession, url: str, token: str = JINA_API_KEY) -> str:
    jina_url = f"https://r.jina.ai/{quote(url)}"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Retain-Images": "none",
        "X-Md-Link-Style": "discarded",
        "X-Timeout": "120",
        "X-Token-Budget": "200000",
        "Authorization": f"Bearer {token}" if token else "",
    }
    try:
        async with session.get(jina_url, headers=headers, timeout=aiohttp.ClientTimeout(total=180)) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("data", {}).get("content", "") or ""
    except Exception as e:
        print(f"[get_url_content] ERROR '{url}': {e}")
        return ""

async def _fetch_one(url: str, session: aiohttp.ClientSession, sem: asyncio.Semaphore, pbar: tqdm = None) -> Dict[str, Any]:
    async with sem:
        content = await get_url_content(session, url)
    if pbar:
        pbar.update(1)
    return {
        "url": url,
        "content": content,
        "fetched_at": datetime.now(tzutc()).isoformat(),
    }

async def fetch_all_urls(unique_urls: List[str], concurrency: int, pbar: tqdm = None) -> Dict[str, Dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=concurrency)
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_one(u, session, sem, pbar) for u in unique_urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return {r["url"]: r for r in results}

def search_and_fetch(nodes: List[Dict[str, Any]], k_per_query: int, concurrency: int) -> List[Dict[str, Any]]:
    all_urls: Set[str] = set()
    query_hits: Dict[str, List[Dict[str, Any]]] = {}

    print("Starting URL search...")
    for node in tqdm(nodes, desc="Searching node queries", ncols=100):
        queries = node.get("queries") or [] 
        for q in queries:
            q_str = (q or "").strip()
            if not q_str:
                continue
            raw_items = get_urls(q_str, topk=k_per_query, token=JINA_API_KEY)
            hits = []
            for it in raw_items[:k_per_query]:
                ext = _extract_url_fields(it)
                url = ext.get("url")
                if not url:
                    continue
                hits.append({"url": url, "title": ext.get("title", ""), "snippet": ext.get("snippet", "")})
                all_urls.add(url)
            query_hits[q_str] = hits

    unique_urls = list(all_urls)
    
    print(f"Starting to fetch content for {len(unique_urls)} URLs...")
    pbar_fetch = tqdm(total=len(unique_urls), desc="Fetching URL content", ncols=100)
    
    async def fetch_with_progress():
        return await fetch_all_urls(unique_urls, concurrency=concurrency, pbar=pbar_fetch)
    
    url_to_doc = asyncio.run(fetch_with_progress()) if unique_urls else {}
    pbar_fetch.close()

    print("Assembling results...")
    enriched_nodes = []
    for node in tqdm(nodes, desc="Assembling node results", ncols=100):
        queries = node.get("queries") or []
        grouped_results = []
        for q in queries:
            hits = query_hits.get(q.strip(), [])
            enriched_hits = []
            for h in hits:
                url = h["url"]
                fetched = url_to_doc.get(url, {})
                enriched_hits.append(
                    {
                        "url": url,
                        "title": h.get("title", ""),
                        "snippet": h.get("snippet", ""),
                        "content": fetched.get("content", ""),
                        "fetched_at": fetched.get("fetched_at", ""),
                    }
                )
            grouped_results.append({"query": q, "hits": enriched_hits})

        enriched_nodes.append(
            {
                "id": node.get("id", ""),
                "topic": node.get("topic"),
                "queries": queries,
                "results": grouped_results,
                "wiki_title": node.get("wiki_title"), 
                "summary": node.get("summary")       
            }
        )
    return enriched_nodes

UTM_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "gclid", "fbclid", "mc_cid", "mc_eid",
}

def normalize_url(u: str) -> str:
    try:
        p = urlparse(u.strip())
        scheme = (p.scheme or "https").lower()
        netloc = p.netloc.lower()
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        qs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in UTM_KEYS]
        qs.sort()
        query = urlencode(qs)
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return u.strip()

def clean_content(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return "\n".join(lines).strip()

def is_valid_content(text: str) -> bool:
    if not text or len(text) < 200:
        return False
    text_lower = text.lower()
    try:
        if detect(text) != 'en':
            return False
    except:
        pass 

    error_keywords = [
        "enable javascript", "please enable cookies", "403 forbidden", "404 not found", 
        "access denied", "verify you are human", "cloudflare", "captcha", 
        "browser is out of date", "terms of service", "privacy policy"
    ]
    if len(text) < 1000:
        for keyword in error_keywords:
            if keyword in text_lower:
                return False
    return True

rc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", ";", ":", "—", "-", " ", ""],
    length_function=len,
)
cap_splitter = TokenTextSplitter(
    chunk_size=TOKEN_CHUNK,
    chunk_overlap=TOKEN_OVERLAP,
    encoding_name="cl100k_base",
    disallowed_special=(),
)

def split_text_hybrid(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    rough = rc_splitter.split_text(text)
    final = []
    for part in rough:
        final.extend(cap_splitter.split_text(part))
    return final

def dedup_and_chunk(enriched_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_norm = {}
    for node in enriched_nodes:
        for group in node.get("results", []):
            for hit in group.get("hits", []):
                u = hit.get("url")
                if not u: continue
                norm = normalize_url(u)
                content = (hit.get("content") or "").strip()
                if not content: continue
                if norm not in best_by_norm or len(content) > len(best_by_norm[norm]["content"]):
                    best_by_norm[norm] = {"url": u, "content": content, "title": hit.get("title")}

    results = []
    seen_doc_hashes = set()
    seen_chunk_hashes = set()
    total_raw_chunks = 0
    total_final_chunks = 0
    
    for idx, (norm, rec) in enumerate(tqdm(best_by_norm.items(), desc="Processing & Chunking", ncols=100), start=1):
        cleaned = clean_content(rec["content"])
        if not is_valid_content(cleaned): continue
        
        doc_hash = hashlib.md5(cleaned.encode('utf-8')).hexdigest()
        if doc_hash in seen_doc_hashes: continue
        seen_doc_hashes.add(doc_hash)
        
        raw_chunks = split_text_hybrid(cleaned)
        if not raw_chunks: continue
        total_raw_chunks += len(raw_chunks)

        unique_chunks_for_this_doc = []
        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text: continue
            chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            if chunk_hash in seen_chunk_hashes: continue
            seen_chunk_hashes.add(chunk_hash)
            unique_chunks_for_this_doc.append(chunk_text)
        
        if not unique_chunks_for_this_doc: continue
        total_final_chunks += len(unique_chunks_for_this_doc)
        
        results.append({
            "id": idx,
            "url": rec["url"],
            "title": rec.get("title", ""),
            "content": cleaned,
            "content_chunk": unique_chunks_for_this_doc
        })

    print(f"\nProcessing complete! Final docs: {len(results)}, Chunks: {total_final_chunks}")
    return results

def build_faiss(chunks: List[Dict[str, Any]], index_path: str, model_path: str):
    lc_embedder = UniversalEmbeddingWrapper(model_path=model_path, device="cuda")

    texts = []
    metadatas = []
    for rec in chunks:
        chunk_content = rec.get("content_chunk", [])
        if isinstance(chunk_content, str):
            chunk_content = [chunk_content]
            
        for chunk_text in chunk_content:
            texts.append(chunk_text)
            metadatas.append({"source_url": rec["url"], "doc_id": rec["id"]})

    if not texts:
        print("Warning: No texts to vectorize.")
        if os.path.exists(index_path):
            print(f"Attempting to load existing index: {index_path}")
            return FAISS.load_local(index_path, lc_embedder, allow_dangerous_deserialization=True)
        return None

    print(f"Extracted {len(texts)} texts, building FAISS index...")
    batch_size = 2000
    vectorstore = None
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Building FAISS Index"):
        batch_texts = texts[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch_texts, lc_embedder, metadatas=batch_metas)
        else:
            vectorstore.add_texts(batch_texts, metadatas=batch_metas)

    if vectorstore:
        print(f"Saving FAISS index to: {index_path}")
        vectorstore.save_local(index_path)
    
    return vectorstore

def retrieve_topk(nodes: List[Dict[str, Any]], vectorstore, k: int = RETURN_TOPK) -> List[Dict[str, Any]]:
    if not vectorstore:
        print("Error: Vectorstore not initialized, cannot retrieve.")
        return []

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    outputs = []
    
    print("Starting to retrieve context for each node...")
    for node in tqdm(nodes, desc="Retrieving", ncols=100):
        queries = node.get("queries") or []
        if not queries:
            continue
        
        import random
        random.seed(42) 
        query = random.choice(queries)
        
        try:
            docs = retriever.get_relevant_documents(query)
            hits = [d.page_content.strip() for d in docs]
            
            outputs.append({
                "id": node.get("id"),
                "topic": node.get("wiki_title"),
                "query": query, 
                "top_k_docs": hits, 
                "summary": node.get("summary")
            })
        except Exception as e:
            print(f"Error retrieving for node {node.get('id')}: {e}")
            
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_path", type=str, default="BAAI/bge-large-en-v1.5")
    
    args = parser.parse_args()

    topic_name_snake_case = args.topic.replace(" ", "_")
    topic_dir = os.path.join(args.data_dir, args.category, topic_name_snake_case)
    
    topic_tree_filename = f"topic_tree_{topic_name_snake_case}_with_summary_queries.json"
    topic_tree_path = os.path.join(topic_dir, topic_tree_filename)
    out_path = os.path.join(topic_dir, f"{topic_name_snake_case}_top50_results.json")
    faiss_index_path = os.path.join(topic_dir, "faiss_index")

    print(f"========== Pipeline Step 0: Context Creation ==========")
    print(f"Category : {args.category}")
    print(f"Topic    : {args.topic}")
    print(f"Work Dir : {topic_dir}")
    print(f"Input    : {topic_tree_path}")
    print(f"Output   : {out_path}")

    if not os.path.exists(topic_dir):
        print(f"Error: Topic directory does not exist -> {topic_dir}")
        return

    if not os.path.exists(topic_tree_path):
        print(f"Error: Input JSON file does not exist -> {topic_tree_path}")
        return

    t0 = time.time()
    
    with open(topic_tree_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    print(f"Read {len(nodes)} nodes")

    enriched = search_and_fetch(nodes, k_per_query=SEARCH_TOPK, concurrency=CONCURRENCY)

    chunks = dedup_and_chunk(enriched)
    
    chunk_save_path = os.path.join(topic_dir, "extracted_passage_chunks.json")
    with open(chunk_save_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    vectorstore = build_faiss(chunks, index_path=faiss_index_path, model_path=args.model_path)

    if vectorstore:
        results = retrieve_topk(nodes, vectorstore, k=RETURN_TOPK)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Done, results saved to: {out_path}")
    else:
        print("Error: Vector library build failed, cannot retrieve.")

    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()