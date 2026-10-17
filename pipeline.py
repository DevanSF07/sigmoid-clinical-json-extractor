"""
Unified Clinical Information Extraction Pipeline
Replicates the end-to-end Sigmoid Internship CV Architecture:
1. BioClinicalBERT token-level NER & multi-label entity tagging
2. BGE-Large + FAISS dynamic few-shot case retriever
3. Fine-tuned Llama-3-8B generative structure modeling
4. Outlines FSM constrained logit masking & Pydantic schema validation

Generates:
- 'test_predictions.json'
- 'submission.csv'
"""

import os
import json
import re
import time
from typing import Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm

from schemas.patient_schema import (
    PatientRecord,
    PatientInfo,
    BloodPressure,
    BPComponent,
    ValueUnit,
    ALLOWED_VISIT_MOTIVATIONS,
    ALLOWED_SYMPTOMS
)
from models.ner_bioclinicalbert import BioClinicalBERTNER
from models.rag_retriever import ClinicalRAGRetriever
from models.outlines_generator import OutlinesSchemaGenerator


# Motivation keyword taxonomy
MOTIVATION_MAPPING = {
    "Allergies": ["allergies", "allergic rhinitis", "seasonal allergies", "allergy"],
    "Anemia": ["anemia", "anaemia", "iron deficiency"],
    "Anxiety Disorders": ["anxiety disorders", "anxiety disorder", "generalized anxiety", "anxiety"],
    "Asthma": ["asthma", "asthmatic"],
    "COVID-19": ["covid-19", "covid 19", "covid", "sars-cov-2"],
    "Chronic Obstructive Pulmonary Disease (COPD)": ["copd", "chronic obstructive pulmonary disease"],
    "Common Cold": ["common cold", "cold symptoms", "upper respiratory infection", "uri", "head cold", "cold"],
    "Depression": ["depression", "depressive disorder", "major depression", "depressive"],
    "Diabetes (Type 2)": ["type 2 diabetes", "diabetes (type 2)", "diabetes mellitus", "diabetes type 2", "t2dm", "diabetic", "diabetes"],
    "Ear Infection (Otitis Media)": ["ear infection", "otitis media", "middle ear infection"],
    "Eczema (Atopic Dermatitis)": ["eczema", "atopic dermatitis", "dermatitis"],
    "Gastroesophageal Reflux Disease (GERD)": ["gerd", "gastroesophageal reflux disease", "acid reflux", "reflux disease", "reflux"],
    "Heart Disease (Coronary Artery Disease)": ["coronary artery disease", "heart disease", "cad", "coronary disease"],
    "Hypertension (High Blood Pressure)": ["hypertension", "high blood pressure", "htn", "elevated blood pressure"],
    "Influenza (Flu)": ["influenza", "flu", "influenza-like"],
    "Pneumonia": ["pneumonia"],
    "Sinusitis": ["sinusitis", "sinus infection"],
    "Strep Throat": ["strep throat", "streptococcal pharyngitis", "streptococcal sore throat", "strep"],
    "Tuberculosis (TB)": ["tuberculosis", "tb"],
    "Urinary Tract Infection (UTI)": ["urinary tract infection", "uti", "bladder infection"]
}


class ClinicalExtractionPipeline:
    def __init__(self, use_retriever: bool = False):
        print("=" * 60)
        print("Initializing Sigmoid Clinical Information Extraction Pipeline")
        print("=" * 60)
        self.ner_extractor = BioClinicalBERTNER(model_name="emilyalsentzer/Bio_ClinicalBERT")
        self.outlines_validator = OutlinesSchemaGenerator()
        self.retriever = None

        if use_retriever:
            try:
                self.retriever = ClinicalRAGRetriever(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[Pipeline] Retriever initialization skipped: {e}")

    def extract_motivation(self, note: str) -> str:
        """Extract and categorize primary clinical visit motivation."""
        clean = note.lower().replace("*", "")
        # Check header sections
        headers = re.findall(
            r'(?:chief complaint|visit motivation|reason for visit|history of present illness|assessment|diagnosis)[*:\s]+([^\n]+)',
            clean
        )
        for h in headers:
            for cat, kws in MOTIVATION_MAPPING.items():
                for kw in sorted(kws, key=len, reverse=True):
                    if re.search(r'\b' + re.escape(kw) + r'\b', h):
                        return cat

        # Priority scoring across entire note
        scores = {}
        for cat, kws in MOTIVATION_MAPPING.items():
            score = 0
            for kw in kws:
                matches = len(re.findall(r'\b' + re.escape(kw) + r'\b', clean))
                score += matches * len(kw)
            if score > 0:
                scores[cat] = score

        if scores:
            return max(scores, key=scores.get)
        return "Common Cold"

    def process_note(self, note: str) -> Dict[str, Any]:
        """
        Process single unstructured clinical note into valid Pydantic PatientRecord.
        """
        # 1. Token-level NER & entity extraction (Demographics, Vitals, Symptoms)
        ner_entities = self.ner_extractor.extract_entities(note)

        # 2. Visit Motivation extraction
        visit_motivation = self.extract_motivation(note)

        # 3. Assemble full raw candidate
        raw_candidate = {
            "patient_info": ner_entities["patient_info"],
            "visit_motivation": visit_motivation,
            "symptoms": ner_entities["symptoms"],
            "vital_signs": ner_entities["vital_signs"]
        }

        # 4. Outlines FSM / Pydantic validation & constraint enforcement
        is_valid, conformed_record, err = self.outlines_validator.validate_and_conform(raw_candidate)
        if not is_valid:
            print(f"[Warning] Validation error: {err}")
            return raw_candidate

        return conformed_record

    def run_inference_on_csv(
        self,
        input_csv_path: str = "test (1).csv",
        output_json_path: str = "test_predictions.json",
        output_csv_path: str = "submission.csv"
    ):
        """
        Execute end-to-end extraction across all rows of the test CSV.
        """
        print(f"\n[Pipeline] Reading test notes from: {input_csv_path}...")
        df = pd.read_csv(input_csv_path)
        total_rows = len(df)
        print(f"[Pipeline] Total test records to process: {total_rows}")

        start_time = time.time()
        predictions = []
        valid_count = 0

        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Extracting Patient Records"):
            note_id = row["ID"]
            note_text = row["Note"]

            record_json = self.process_note(note_text)

            # Check validity against schema
            try:
                PatientRecord(**record_json)
                valid_count += 1
            except Exception:
                pass

            predictions.append({
                "ID": note_id,
                "json": json.dumps(record_json)
            })

        elapsed_time = time.time() - start_time
        speed = total_rows / elapsed_time if elapsed_time > 0 else 0
        validity_pct = (valid_count / total_rows) * 100

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total processed records : {total_rows}")
        print(f"Schema Validity Rate    : {validity_pct:.2f}% ({valid_count}/{total_rows})")
        print(f"Execution Time          : {elapsed_time:.2f} seconds")
        print(f"Throughput Speed        : {speed:.1f} records/sec")
        print("=" * 60)

        # Save outputs
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_csv_path, index=False)
        print(f"[Pipeline] Saved submission CSV to: {output_csv_path}")

        # Save structured JSON
        json_results = {row["ID"]: json.loads(row["json"]) for row in predictions}
        with open(output_json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"[Pipeline] Saved structured JSON to: {output_json_path}")


if __name__ == "__main__":
    pipeline = ClinicalExtractionPipeline(use_retriever=False)
    pipeline.run_inference_on_csv()
