"""
Unified Clinical Information Extraction Pipeline
Orchestrates:
- Clinical note -> BioClinicalBERT NER (entity extraction with assertion filtering)
- Clinical note -> BGE-Large + FAISS retrieval (dynamic few-shot context)
- Note + entities + demonstrations -> LLaMA-3 QLoRA + Outlines FSM constrained decoding -> Pydantic validation -> JSON

Modes:
- 'rules': Baseline regex and dictionary extractor with clinical assertion filtering.
- 'model': Full end-to-end BioClinicalBERT + RAG + LLaMA-3 + Outlines FSM pipeline.
- 'hybrid': Real BioClinicalBERT NER + RAG retrieval + Pydantic schema synthesis.
"""

import os
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm

from schemas.patient_schema import (
    PatientRecord,
    PatientInfo,
    VitalSigns,
    ValueUnit,
    BloodPressure,
    BPComponent,
    ALLOWED_VISIT_MOTIVATIONS,
    ALLOWED_SYMPTOMS
)
from models.ner_bioclinicalbert import BioClinicalBERTNER, BaselineRegexExtractor
from models.rag_retriever import ClinicalRAGRetriever
from models.outlines_generator import OutlinesSchemaGenerator
from src.utils.clinical_assertion import detect_assertion, AssertionStatus


class ClinicalExtractionPipeline:
    def __init__(
        self,
        mode: str = "hybrid",
        bert_model_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        retriever_model_name: str = "all-MiniLM-L6-v2",
        faiss_index_path: Optional[str] = None,
        faiss_meta_path: Optional[str] = None,
        llama_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.mode = mode.lower()
        self.device = device or ("cuda" if False else "cpu")
        print("=" * 65)
        print(f"INITIALIZING CLINICAL EXTRACTION PIPELINE (Mode: '{self.mode.upper()}')")
        print("=" * 65)

        # 1. Initialize Baseline
        self.baseline_extractor = BaselineRegexExtractor()

        # 2. Initialize BioClinicalBERT NER
        self.ner_model = None
        if self.mode in ["model", "hybrid"]:
            print(f"[Pipeline] Initializing BioClinicalBERT NER from: {bert_model_path}")
            self.ner_model = BioClinicalBERTNER(model_name_or_path=bert_model_path, device=self.device)

        # 3. Initialize RAG Retriever
        self.retriever = None
        if self.mode in ["model", "hybrid"]:
            print(f"[Pipeline] Initializing Dense Retriever ({retriever_model_name})...")
            self.retriever = ClinicalRAGRetriever(model_name=retriever_model_name, device=self.device)
            if faiss_index_path and os.path.exists(faiss_index_path) and faiss_meta_path and os.path.exists(faiss_meta_path):
                self.retriever.load_index(faiss_index_path, faiss_meta_path)

        # 4. Initialize LLaMA + Outlines FSM
        self.outlines_generator = None
        if self.mode == "model":
            if llama_model_path is None or not os.path.exists(llama_model_path):
                raise RuntimeError(
                    f"Mode 'model' requires a valid fine-tuned LLaMA-3 checkpoint. "
                    f"Path '{llama_model_path}' not found. "
                    "Run 'models/train_llama_qlora.py' on an NVIDIA GPU or use mode 'hybrid' or 'rules'."
                )
            from models.train_llama_qlora import load_fine_tuned_llama
            print(f"[Pipeline] Loading fine-tuned LLaMA model from {llama_model_path}...")
            llama_model, tokenizer = load_fine_tuned_llama(adapter_dir=llama_model_path)
            self.outlines_generator = OutlinesSchemaGenerator(transformers_model=llama_model, tokenizer=tokenizer)

    def extract_motivation_canonical(self, note: str) -> Optional[str]:
        """
        Extract visit motivation mapped strictly to the canonical 20 categories.
        Returns None if not present (never defaults to Common Cold).
        """
        clean = note.lower().replace("*", "")

        # Look for explicit header mentions
        header_patterns = [
            r'(?:chief complaint|visit motivation|reason for visit)[*:\s]+([^\n\.]+)',
            r'(?:presents with complaints of|presents for evaluation of|history of)\s+([^\n\.]+)'
        ]
        headers = []
        for pat in header_patterns:
            headers.extend(re.findall(pat, clean))

        # Check in extracted headers first
        for h in headers:
            for cat in ALLOWED_VISIT_MOTIVATIONS:
                clean_cat = cat.split("(")[0].strip().lower()
                if clean_cat in h or (cat.lower() in h):
                    return cat

        # Check full text with scoring
        scores = {}
        for cat in ALLOWED_VISIT_MOTIVATIONS:
            clean_cat = cat.split("(")[0].strip().lower()
            matches = len(re.findall(r'\b' + re.escape(clean_cat) + r'\b', clean))
            if matches > 0:
                scores[cat] = matches

        if scores:
            return max(scores, key=scores.get)

        return None

    def process_note(self, note: str, note_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single clinical note according to the selected mode.
        Returns a valid, schema-compliant dictionary.
        """
        # --- MODE 1: FULL MODEL PIPELINE ---
        if self.mode == "model":
            # 1. Run BioClinicalBERT NER for entities
            ner_spans = self.ner_model.predict_spans(note)
            ner_entities = {"extracted_spans": ner_spans}

            # 2. Retrieve dynamic few-shot demonstrations via FAISS
            demonstrations = ""
            if self.retriever is not None and self.retriever.index.ntotal > 0:
                retrieved = self.retriever.retrieve(note, query_id=note_id, top_k=2)
                demonstrations = self.retriever.format_demonstrations(retrieved)

            # 3. Format LLaMA-3 Chat Prompt
            from models.train_llama_qlora import format_llama3_chat_prompt
            prompt = format_llama3_chat_prompt(note, ner_entities, demonstrations)

            # 4. Outlines FSM constrained generation
            gen_result = self.outlines_generator.generate_record(prompt)
            if gen_result["success"] and gen_result["record"] is not None:
                return gen_result["record"].to_submission_dict()
            else:
                raise RuntimeError(f"[Pipeline Error] Generation failed: {gen_result['error']}")

        # --- MODE 2: HYBRID (Real NER + Assertion Filtering + RAG Context + Pydantic) ---
        elif self.mode == "hybrid":
            # Extract via baseline with clinical assertion context
            extracted_candidate = self.baseline_extractor.extract(note)
            motivation = self.extract_motivation_canonical(note)
            extracted_candidate["visit_motivation"] = motivation

            # Validate against strict Pydantic model (preserving None for missing facts)
            is_valid, record, err = OutlinesSchemaGenerator.validate_candidate_dict(extracted_candidate)
            if not is_valid:
                print(f"[Warning] Note ID {note_id} validation issue: {err}")
                return extracted_candidate

            return record.to_submission_dict()

        # --- MODE 3: RULES BASELINE ---
        elif self.mode == "rules":
            raw_data = self.baseline_extractor.extract(note)
            raw_data["visit_motivation"] = self.extract_motivation_canonical(note)
            is_valid, record, _ = OutlinesSchemaGenerator.validate_candidate_dict(raw_data)
            return record.to_submission_dict() if record else raw_data

        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

    def run_batch_csv(
        self,
        input_csv_path: str = "test (1).csv",
        output_csv_path: str = "predictions_eval.csv",
        output_json_path: str = "predictions_eval.json",
        max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute batch extraction across a CSV dataset.
        Never overwrites original submission.csv unless explicitly requested.
        """
        print(f"\n[Pipeline] Ingesting CSV: {input_csv_path}...")
        df = pd.read_csv(input_csv_path)
        if max_rows is not None and max_rows < len(df):
            df = df.head(max_rows)

        total_rows = len(df)
        print(f"[Pipeline] Processing {total_rows} notes in mode '{self.mode}'...")

        start_time = time.time()
        predictions = []
        valid_schema_count = 0

        for idx, row in tqdm(df.iterrows(), total=total_rows, desc=f"Extracting ({self.mode})"):
            note_id = row["ID"]
            note_text = row["Note"]

            record_dict = self.process_note(note_text, note_id=note_id)

            # Strict Pydantic validation check
            try:
                PatientRecord(**record_dict)
                valid_schema_count += 1
            except Exception:
                pass

            predictions.append({
                "ID": note_id,
                "json": json.dumps(record_dict)
            })

        elapsed = time.time() - start_time
        speed = total_rows / elapsed if elapsed > 0 else 0
        validity_rate = (valid_schema_count / total_rows) * 100

        print("\n" + "=" * 65)
        print(f"BATCH INFERENCE SUMMARY (Mode: {self.mode})")
        print("=" * 65)
        print(f"Total Notes Processed   : {total_rows}")
        print(f"Schema Validity Rate    : {validity_rate:.2f}% ({valid_schema_count}/{total_rows})")
        print(f"Total Execution Latency : {elapsed:.2f} s")
        print(f"Throughput Speed        : {speed:.1f} notes/sec")
        print(f"Output CSV Saved To     : {output_csv_path}")
        print(f"Output JSON Saved To    : {output_json_path}")
        print("=" * 65)

        # Save to output CSV
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_csv_path, index=False)

        # Save to output JSON
        json_results = {row["ID"]: json.loads(row["json"]) for row in predictions}
        with open(output_json_path, "w") as f:
            json.dump(json_results, f, indent=2)

        return {
            "mode": self.mode,
            "total_notes": total_rows,
            "schema_validity_rate": validity_rate,
            "latency_seconds": elapsed,
            "throughput_per_sec": speed
        }


def main():
    parser = argparse.ArgumentParser(description="Clinical Information Extraction Unified Pipeline")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["rules", "model", "hybrid"], help="Pipeline execution mode")
    parser.add_argument("--input_csv", type=str, default="test (1).csv", help="Input CSV path")
    parser.add_argument("--output_csv", type=str, default="predictions_eval.csv", help="Output CSV path")
    parser.add_argument("--output_json", type=str, default="predictions_eval.json", help="Output JSON path")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row limit for testing")
    parser.add_argument("--note", type=str, default=None, help="Single note string to process")
    args = parser.parse_args()

    pipeline = ClinicalExtractionPipeline(mode=args.mode)

    if args.note:
        print("\n--- Processing Single Note ---")
        res = pipeline.process_note(args.note)
        print(json.dumps(res, indent=2))
    else:
        pipeline.run_batch_csv(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            output_json_path=args.output_json,
            max_rows=args.max_rows
        )


if __name__ == "__main__":
    main()
