"""
RAG Retriever Module: BGE-Large + FAISS
Retrieves semantically similar clinical cases from historical notes
to provide dynamic few-shot context for structured JSON generation.

CV Metric Targets:
- 0.89 MRR (Mean Reciprocal Rank)
- 93.1% Hit@3
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer


class ClinicalRAGRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu"
    ):
        """
        Initialize the BGE-Large dense retriever with FAISS index.
        Falls back to lightweight/available transformer if running locally.
        """
        self.model_name = model_name
        self.device = device
        print(f"[RAG] Loading sentence embedder: {model_name} on {device}...")
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"[RAG] Fallback to 'all-MiniLM-L6-v2' due to download/memory limit: {e}")
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
        self.dimension = self.model.get_sentence_embedding_dimension()
        # FAISS inner product index for normalized vectors (cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.corpus_ids: List[int] = []
        self.corpus_notes: List[str] = []
        self.corpus_jsons: List[Dict[str, Any]] = []

    def build_index(self, train_df: pd.DataFrame, batch_size: int = 64):
        """
        Index train clinical notes into FAISS vector store.
        """
        print(f"[RAG] Encoding {len(train_df)} corpus clinical notes...")
        self.corpus_ids = train_df["ID"].tolist()
        self.corpus_notes = train_df["Note"].tolist()
        self.corpus_jsons = [
            json.loads(j) if isinstance(j, str) else j
            for j in train_df["json"].tolist()
        ]

        # Generate embeddings
        embeddings = self.model.encode(
            self.corpus_notes,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index
        self.index.reset()
        self.index.add(embeddings)
        print(f"[RAG] Successfully indexed {self.index.ntotal} records into FAISS.")

    def retrieve(self, query_note: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar clinical cases for dynamic few-shot prompting.
        """
        query_embedding = self.model.encode(
            [query_note],
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.corpus_notes):
                continue
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "id": self.corpus_ids[idx],
                "note": self.corpus_notes[idx],
                "json": self.corpus_jsons[idx]
            })
        return results

    def format_few_shot_prompt(self, retrieved_cases: List[Dict[str, Any]], query_note: str) -> str:
        """
        Construct dynamic few-shot prompt for Llama-3-8B JSON generation.
        """
        prompt = (
            "You are an expert clinical informatics model. Your task is to extract "
            "structured clinical records from unstructured medical notes into valid JSON.\n\n"
        )
        for i, case in enumerate(retrieved_cases, 1):
            prompt += f"### Example Case {i}:\n"
            prompt += f"**Clinical Note:**\n{case['note'].strip()}\n\n"
            prompt += f"**Target Structured JSON:**\n{json.dumps(case['json'])}\n\n"

        prompt += "### Target Note To Process:\n"
        prompt += f"**Clinical Note:**\n{query_note.strip()}\n\n"
        prompt += "**Target Structured JSON:**\n"
        return prompt

    def evaluate_retrieval(
        self,
        eval_df: pd.DataFrame,
        top_k: int = 3,
        sample_size: int = 500
    ) -> Dict[str, float]:
        """
        Evaluate Hit@K and MRR based on visit motivation match.
        """
        eval_samples = eval_df.sample(min(sample_size, len(eval_df)), random_state=42)
        hits_at_k = 0
        reciprocal_ranks = []

        query_embeddings = self.model.encode(
            eval_samples["Note"].tolist(),
            batch_size=64,
            normalize_embeddings=True
        )
        query_embeddings = np.array(query_embeddings, dtype=np.float32)

        scores, indices = self.index.search(query_embeddings, top_k + 1)

        for i, (_, row) in enumerate(eval_samples.iterrows()):
            true_json = json.loads(row["json"]) if isinstance(row["json"], str) else row["json"]
            true_mot = true_json.get("visit_motivation", "")

            # Exclude self-match (index corresponding to exact same ID)
            retrieved_matches = []
            for rank_idx in indices[i]:
                if self.corpus_ids[rank_idx] != row["ID"]:
                    retrieved_matches.append(rank_idx)
                if len(retrieved_matches) == top_k:
                    break

            # Check if any retrieved sample has the same visit motivation
            found_rank = None
            for r, match_idx in enumerate(retrieved_matches, 1):
                cand_mot = self.corpus_jsons[match_idx].get("visit_motivation", "")
                if cand_mot == true_mot:
                    found_rank = r
                    break

            if found_rank is not None:
                hits_at_k += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

        hit_rate = hits_at_k / len(eval_samples)
        mrr = float(np.mean(reciprocal_ranks))

        print(f"[RAG Eval] Hit@{top_k}: {hit_rate * 100:.2f}% | MRR: {mrr:.4f}")
        return {
            f"Hit@{top_k}": hit_rate,
            "MRR": mrr
        }
