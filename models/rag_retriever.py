"""
BGE-Large Dense Retrieval & FAISS Index Module
Indexes clinical cases strictly from the training split.
Retrieves semantically relevant few-shot in-context demonstrations for LLaMA-3.
Enforces query self-exclusion, metadata serialization, and honest proxy evaluation.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


class ClinicalRAGRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        max_seq_length: int = 512
    ):
        """
        Initialize dense retriever.
        Model selection is explicit in configuration and logged.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if False else "cpu")
        self.max_seq_length = max_seq_length

        print(f"[RAG Retriever] Initializing embedding model: '{self.model_name}' on device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = self.max_seq_length

        self.dimension = self.model.get_sentence_embedding_dimension()
        # FAISS inner product index over L2-normalized vectors (exact cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)

        self.corpus_ids: List[int] = []
        self.corpus_notes: List[str] = []
        self.corpus_jsons: List[Dict[str, Any]] = []

    def build_index(
        self,
        train_df: pd.DataFrame,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Encode training notes and populate FAISS index.
        Index must be built strictly from training split data.
        """
        print(f"[RAG Retriever] Building FAISS index from {len(train_df)} training records...")
        self.corpus_ids = train_df["ID"].tolist()
        self.corpus_notes = train_df["Note"].tolist()
        self.corpus_jsons = [
            json.loads(j) if isinstance(j, str) else j
            for j in train_df["json"].tolist()
        ]

        # BGE input convention for retrieval
        embeddings = self.model.encode(
            self.corpus_notes,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.reset()
        self.index.add(embeddings)
        print(f"[RAG Retriever] Successfully indexed {self.index.ntotal} vectors (dimension={self.dimension}).")

    def save_index(self, index_path: str = "checkpoints/faiss_index.bin", metadata_path: str = "checkpoints/faiss_metadata.json"):
        """Save FAISS binary index and metadata."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        metadata = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "total_records": len(self.corpus_ids),
            "corpus_ids": self.corpus_ids,
            "corpus_notes": self.corpus_notes,
            "corpus_jsons": self.corpus_jsons
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[RAG Retriever] Index and metadata saved to {index_path} and {metadata_path}")

    def load_index(self, index_path: str = "checkpoints/faiss_index.bin", metadata_path: str = "checkpoints/faiss_metadata.json"):
        """Load saved FAISS binary index and metadata."""
        print(f"[RAG Retriever] Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.corpus_ids = metadata["corpus_ids"]
        self.corpus_notes = metadata["corpus_notes"]
        self.corpus_jsons = metadata["corpus_jsons"]
        print(f"[RAG Retriever] Loaded {self.index.ntotal} vectors from index.")

    def retrieve(
        self,
        query_note: str,
        query_id: Optional[int] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar training cases.
        Enforces self-exclusion: excludes the query note itself and identical duplicates.
        """
        query_embedding = self.model.encode([query_note], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Retrieve extra candidates to account for self-exclusion
        search_k = min(top_k + 5, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for rank_pos, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.corpus_ids):
                continue

            cand_id = self.corpus_ids[idx]
            cand_note = self.corpus_notes[idx]

            # Self-exclusion and duplicate check
            if query_id is not None and cand_id == query_id:
                continue
            if cand_note.strip() == query_note.strip():
                continue

            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "id": cand_id,
                "note": cand_note,
                "json": self.corpus_jsons[idx]
            })

            if len(results) == top_k:
                break

        return results

    def format_demonstrations(self, retrieved_cases: List[Dict[str, Any]]) -> str:
        """Format retrieved cases into few-shot demonstration context."""
        demo_str = ""
        for i, case in enumerate(retrieved_cases, 1):
            clean_json = json.dumps(case["json"])
            demo_str += f"### Example {i} (Clinical Case):\n"
            demo_str += f"**Clinical Note:**\n{case['note'].strip()}\n"
            demo_str += f"**Validated Structured JSON:**\n{clean_json}\n\n"
        return demo_str

    def evaluate_retrieval(
        self,
        eval_df: pd.DataFrame,
        top_k: int = 3,
        sample_size: int = 300
    ) -> Dict[str, Any]:
        """
        Evaluate Hit@K and MRR@K against evaluation dataset.
        Relevance proxy: Category agreement on visit_motivation.
        """
        eval_samples = eval_df.sample(min(sample_size, len(eval_df)), random_state=42).reset_index(drop=True)
        hits = 0
        reciprocal_ranks = []

        print(f"[RAG Eval] Evaluating Hit@{top_k} and MRR@{top_k} on {len(eval_samples)} cases...")
        for _, row in eval_samples.iterrows():
            q_id = row["ID"]
            q_note = row["Note"]
            true_json = json.loads(row["json"]) if isinstance(row["json"], str) else row["json"]
            true_mot = true_json.get("visit_motivation")

            retrieved = self.retrieve(q_note, query_id=q_id, top_k=top_k)

            found_rank = None
            for case in retrieved:
                cand_mot = case["json"].get("visit_motivation")
                if cand_mot == true_mot and true_mot is not None:
                    found_rank = case["rank"]
                    break

            if found_rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

        hit_rate = hits / len(eval_samples) if eval_samples is not None else 0.0
        mrr = float(np.mean(reciprocal_ranks))

        results = {
            "model_name": self.model_name,
            "evaluated_samples": len(eval_samples),
            "relevance_definition": "Same visit_motivation (category agreement proxy)",
            f"Hit@{top_k}": round(hit_rate, 4),
            f"MRR@{top_k}": round(mrr, 4)
        }
        print(f"[RAG Eval Results] Hit@{top_k}: {hit_rate*100:.2f}% | MRR@{top_k}: {mrr:.4f} ({results['relevance_definition']})")
        return results
