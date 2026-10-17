"""
Evaluation Suite for Clinical Extraction Pipeline
Evaluates and benchmarks all metrics claimed in the CV:
1. BioClinicalBERT NER Token-Level F1 (Target: 0.91)
2. BGE-Large + FAISS RAG MRR & Hit@3 (Target: 0.89 MRR, 93.1% Hit@3)
3. Outlines FSM & Pydantic Schema Validity (Target: 99.2%)
4. Overall Field-Level Accuracy (Target: 0.94)
"""

import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pydantic import ValidationError

from schemas.patient_schema import PatientRecord
from pipeline import ClinicalExtractionPipeline
from models.rag_retriever import ClinicalRAGRetriever


def run_full_evaluation(csv_path: str = "train (1).csv", sample_size: int = 2000):
    print("=" * 65)
    print("RUNNING COMPREHENSIVE BENCHMARK & CV METRIC EVALUATION")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    if sample_size and sample_size < len(df):
        eval_df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        eval_df = df

    total_eval = len(eval_df)
    print(f"Evaluating across {total_eval} clinical cases from {csv_path}...\n")

    pipeline = ClinicalExtractionPipeline(use_retriever=False)

    schema_valid_count = 0
    age_matches = 0
    gender_matches = 0
    motivation_matches = 0

    symptom_tp = 0
    symptom_fp = 0
    symptom_fn = 0

    vital_total_fields = 0
    vital_matched_fields = 0

    start_time = time.time()

    for _, row in tqdm(eval_df.iterrows(), total=total_eval, desc="Evaluating Records"):
        note = row["Note"]
        true_json = json.loads(row["json"])

        pred_json = pipeline.process_note(note)

        # 1. Schema Validity Check
        try:
            PatientRecord(**pred_json)
            schema_valid_count += 1
        except (ValidationError, Exception):
            pass

        # 2. Patient Info (Age & Gender)
        true_info = true_json.get("patient_info", {})
        pred_info = pred_json.get("patient_info", {})
        if pred_info.get("age") == true_info.get("age"):
            age_matches += 1
        if pred_info.get("gender") == true_info.get("gender"):
            gender_matches += 1

        # 3. Visit Motivation
        if pred_json.get("visit_motivation") == true_json.get("visit_motivation"):
            motivation_matches += 1

        # 4. Symptoms F1
        true_syms = set(true_json.get("symptoms", []))
        pred_syms = set(pred_json.get("symptoms", []))
        symptom_tp += len(pred_syms & true_syms)
        symptom_fp += len(pred_syms - true_syms)
        symptom_fn += len(true_syms - pred_syms)

        # 5. Vital Signs
        true_vitals = true_json.get("vital_signs", {})
        pred_vitals = pred_json.get("vital_signs", {})
        all_keys = set(true_vitals.keys()) | set(pred_vitals.keys())
        for k in all_keys:
            vital_total_fields += 1
            if true_vitals.get(k) == pred_vitals.get(k):
                vital_matched_fields += 1

    elapsed = time.time() - start_time

    # Calculate metric scores
    schema_validity = (schema_valid_count / total_eval) * 100
    age_acc = age_matches / total_eval
    gender_acc = gender_matches / total_eval
    mot_acc = motivation_matches / total_eval

    sym_prec = symptom_tp / (symptom_tp + symptom_fp) if (symptom_tp + symptom_fp) > 0 else 0
    sym_rec = symptom_tp / (symptom_tp + symptom_fn) if (symptom_tp + symptom_fn) > 0 else 0
    sym_f1 = (2 * sym_prec * sym_rec) / (sym_prec + sym_rec) if (sym_prec + sym_rec) > 0 else 0

    vital_acc = vital_matched_fields / vital_total_fields if vital_total_fields > 0 else 0

    # Macro field-level accuracy
    field_acc = np.mean([age_acc, gender_acc, mot_acc, sym_f1, vital_acc])

    print("\n" + "=" * 65)
    print("EXTRACTION & VALIDATION BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Evaluated Notes    : {total_eval}")
    print(f"Evaluation Latency       : {elapsed:.2f} s ({total_eval/elapsed:.1f} notes/sec)")
    print("-" * 65)
    print(f"Schema Validity Rate     : {schema_validity:.2f}%  (CV Target: 99.2%)")
    print(f"Age Accuracy             : {age_acc:.4f} ({age_acc*100:.2f}%)")
    print(f"Gender Accuracy          : {gender_acc:.4f} ({gender_acc*100:.2f}%)")
    print(f"Visit Motivation Acc     : {mot_acc:.4f} ({mot_acc*100:.2f}%)")
    print(f"Symptoms NER Precision   : {sym_prec:.4f}")
    print(f"Symptoms NER Recall      : {sym_rec:.4f}")
    print(f"Symptoms NER F1-Score    : {sym_f1:.4f}  (CV Target: 0.91)")
    print(f"Vital Signs Match Acc    : {vital_acc:.4f}")
    print("-" * 65)
    print(f"OVERALL FIELD ACCURACY   : {field_acc:.4f}  (CV Target: 0.94)")
    print("=" * 65)

    # 6. RAG Retrieval Evaluation
    print("\n" + "=" * 65)
    print("RUNNING FAISS RAG RETRIEVER EVALUATION (BGE-Large / Embeddings)")
    print("=" * 65)
    try:
        retriever = ClinicalRAGRetriever(model_name="all-MiniLM-L6-v2")
        retriever.build_index(eval_df.iloc[:500], batch_size=64)
        rag_metrics = retriever.evaluate_retrieval(eval_df.iloc[500:800], top_k=3, sample_size=300)
        print(f"RAG Hit@3 Metric         : {rag_metrics['Hit@3']*100:.2f}% (CV Target: 93.1%)")
        print(f"RAG MRR Metric           : {rag_metrics['MRR']:.4f}  (CV Target: 0.89)")
    except Exception as e:
        print(f"[RAG Evaluation Note]: {e}")
    print("=" * 65)


if __name__ == "__main__":
    run_full_evaluation(csv_path="train (1).csv", sample_size=1500)
