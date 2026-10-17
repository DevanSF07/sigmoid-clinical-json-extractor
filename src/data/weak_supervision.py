"""
Weak Supervision Span Alignment and Provenance Module
Aligns document-level JSON annotations with clinical note text spans.
Generates token-level BIO datasets, provenance tracking, and gold manual review format.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from tqdm import tqdm

from src.utils.clinical_assertion import detect_assertion, AssertionStatus


# Vocabulary aliases for symptoms
SYMPTOM_ALIASES: Dict[str, List[str]] = {
    "abdominal_pain": ["abdominal pain", "stomach pain", "belly pain", "abdominal discomfort", "pain in abdomen"],
    "anxiety": ["anxiety", "anxious", "feeling anxious", "generalized anxiety", "anxiety symptoms"],
    "blurred_vision": ["blurred vision", "blurry vision", "vision blurriness"],
    "chest_pain": ["chest pain", "pain in chest", "substernal chest pain"],
    "cough": ["cough", "coughing", "productive cough", "dry cough"],
    "diarrhea": ["diarrhea", "loose stools", "frequent loose stools"],
    "difficulty_breathing": ["difficulty breathing", "shortness of breath", "dyspnea", "labored breathing", "breathlessness"],
    "difficulty_concentrating": ["difficulty concentrating", "trouble concentrating", "poor concentration", "inability to concentrate"],
    "dizziness": ["dizziness", "lightheadedness", "feeling dizzy", "lightheaded"],
    "dry_skin": ["dry skin", "skin dryness", "xerosis"],
    "ear_pain": ["ear pain", "earache", "pain in ear", "otalgia"],
    "facial_pain": ["facial pain", "pain in face", "sinus facial pain"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "lethargy", "feeling fatigued"],
    "fever": ["fever", "febrile", "elevated temperature", "high fever", "feverish"],
    "frequent_urination": ["frequent urination", "urinary frequency", "urinating often", "increased urinary frequency"],
    "headache": ["headache", "head pain", "cephalea"],
    "heartburn": ["heartburn", "acid reflux", "pyrosis", "acid indigestion"],
    "increased_thirst": ["increased thirst", "excessive thirst", "polydipsia", "feeling thirsty"],
    "itchy_eyes": ["itchy eyes", "eye itching", "pruritus of eyes"],
    "joint_pain": ["joint pain", "arthralgia", "pain in joints"],
    "loss_of_taste_smell": ["loss of taste", "loss of smell", "loss of taste and smell", "loss of taste/smell", "anosmia", "ageusia"],
    "nausea": ["nausea", "feeling nauseous", "queasy"],
    "night_sweats": ["night sweats", "sweating at night", "nocturnal diaphoresis"],
    "painful_urination": ["painful urination", "dysuria", "pain during urination", "burning urination"],
    "pale_skin": ["pale skin", "pallor", "pale appearance"],
    "rash": ["rash", "skin rash", "erythematous rash", "skin eruption"],
    "restlessness": ["restlessness", "restless", "feeling restless"],
    "runny_nose": ["runny nose", "rhinorrhea", "nasal discharge", "clear discharge"],
    "sadness": ["sadness", "feeling sad", "depressed mood", "feelings of sadness"],
    "sneezing": ["sneezing", "frequent sneezing", "sneeze"],
    "sore_throat": ["sore throat", "pharyngitis", "throat pain"],
    "swollen_lymph_nodes": ["swollen lymph nodes", "lymphadenopathy", "enlarged lymph nodes"],
    "vomiting": ["vomiting", "emesis", "vomited"],
    "weight_loss": ["weight loss", "unintentional weight loss", "unexplained weight loss"],
    "wheezing": ["wheezing", "wheeze", "audible wheeze"]
}


class WeakSupervisionAligner:
    def __init__(self):
        self.symptom_aliases = SYMPTOM_ALIASES

    def find_spans_for_note(
        self,
        note: str,
        target_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Align structured fields from target_json to exact character spans in note.
        Handles assertions (AFFIRMED vs NEGATED) and disambiguation.
        """
        aligned_spans: List[Dict[str, Any]] = []
        ambiguous_spans: List[Dict[str, Any]] = []
        negated_spans: List[Dict[str, Any]] = []

        # 1. Demographics alignment
        p_info = target_json.get("patient_info", {})
        age = p_info.get("age")
        gender = p_info.get("gender")

        if age is not None:
            # Search for age patterns matching this exact age
            age_patterns = [
                rf"\b{age}\s*[- ]\s*year\s*[- ]\s*old\b",
                rf"\bAge\s*[:\*\s]+\s*{age}\b",
                rf"\bage\s+{age}\b"
            ]
            for pat in age_patterns:
                for m in re.finditer(pat, note, re.I):
                    aligned_spans.append({
                        "start": m.start(),
                        "end": m.end(),
                        "text": note[m.start():m.end()],
                        "label": "PATIENT_INFO",
                        "subtype": "age",
                        "assertion": AssertionStatus.AFFIRMED.value,
                        "confidence": 1.0
                    })
                    break

        if gender:
            gender_patterns = [
                rf"\bGender\s*[:\*\s]+\s*{gender}\b",
                rf"\b{gender}\b"
            ]
            for pat in gender_patterns:
                for m in re.finditer(pat, note, re.I):
                    aligned_spans.append({
                        "start": m.start(),
                        "end": m.end(),
                        "text": note[m.start():m.end()],
                        "label": "PATIENT_INFO",
                        "subtype": "gender",
                        "assertion": AssertionStatus.AFFIRMED.value,
                        "confidence": 1.0
                    })
                    break

        # 2. Vital signs alignment
        vitals = target_json.get("vital_signs", {})
        for vital_name, vital_data in vitals.items():
            if vital_name == "blood_pressure":
                sys_v = vital_data.get("systolic", {}).get("value")
                dia_v = vital_data.get("diastolic", {}).get("value")
                if sys_v and dia_v:
                    bp_pat = rf"\b{sys_v}\s*/\s*{dia_v}\s*mmHg\b"
                    for m in re.finditer(bp_pat, note, re.I):
                        aligned_spans.append({
                            "start": m.start(),
                            "end": m.end(),
                            "text": note[m.start():m.end()],
                            "label": "VITAL_SIGN",
                            "subtype": "blood_pressure",
                            "assertion": AssertionStatus.AFFIRMED.value,
                            "confidence": 1.0
                        })
                        break
            else:
                v_num = vital_data.get("value")
                if v_num is not None:
                    # Look for number followed by unit or nearby
                    num_str = str(int(v_num)) if (isinstance(v_num, float) and v_num.is_integer()) else str(v_num)
                    val_pat = rf"\b{re.escape(num_str)}\b"
                    for m in re.finditer(val_pat, note):
                        aligned_spans.append({
                            "start": m.start(),
                            "end": m.end(),
                            "text": note[m.start():m.end()],
                            "label": "VITAL_SIGN",
                            "subtype": vital_name,
                            "assertion": AssertionStatus.AFFIRMED.value,
                            "confidence": 0.95
                        })
                        break

        # 3. Symptoms alignment with Context Assertion
        target_symptoms = set(target_json.get("symptoms", []))
        aligned_symptoms = set()

        for canonical_sym, aliases in self.symptom_aliases.items():
            for alias in aliases:
                pattern = rf"\b{re.escape(alias)}\b"
                for m in re.finditer(pattern, note, re.I):
                    start, end = m.start(), m.end()
                    span_text = note[start:end]
                    assertion, trig = detect_assertion(note, start, end)

                    span_entry = {
                        "start": start,
                        "end": end,
                        "text": span_text,
                        "canonical": canonical_sym,
                        "label": "SYMPTOM",
                        "assertion": assertion.value,
                        "trigger": trig
                    }

                    if assertion == AssertionStatus.NEGATED:
                        negated_spans.append(span_entry)
                    elif canonical_sym in target_symptoms and assertion == AssertionStatus.AFFIRMED:
                        span_entry["confidence"] = 1.0
                        aligned_spans.append(span_entry)
                        aligned_symptoms.add(canonical_sym)
                    elif canonical_sym not in target_symptoms:
                        span_entry["confidence"] = 0.5
                        ambiguous_spans.append(span_entry)

        # 4. Visit motivation alignment
        motivation = target_json.get("visit_motivation")
        if motivation:
            clean_mot = motivation.split("(")[0].strip()
            for m in re.finditer(rf"\b{re.escape(clean_mot)}\b", note, re.I):
                aligned_spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "text": note[m.start():m.end()],
                    "label": "VISIT_MOTIVATION",
                    "subtype": motivation,
                    "assertion": AssertionStatus.AFFIRMED.value,
                    "confidence": 1.0
                })
                break

        # Calculate coverage for this note
        symptom_coverage = len(aligned_symptoms) / len(target_symptoms) if target_symptoms else 1.0

        return {
            "aligned_spans": aligned_spans,
            "ambiguous_spans": ambiguous_spans,
            "negated_spans": negated_spans,
            "target_symptoms_count": len(target_symptoms),
            "aligned_symptoms_count": len(aligned_symptoms),
            "symptom_coverage": symptom_coverage
        }

    def generate_weakly_supervised_dataset(
        self,
        df: pd.DataFrame,
        output_provenance_path: str = "splits/weak_supervision_provenance.json",
        output_gold_template_path: str = "splits/manual_review_gold_template.json",
        sample_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process dataframe and output span annotations with provenance and review template.
        """
        records_to_process = df.head(sample_limit) if sample_limit else df
        dataset_annotations = []

        total_target_syms = 0
        total_aligned_syms = 0
        total_negated_found = 0
        total_ambiguous_found = 0

        manual_review_samples = []

        print(f"[Weak Supervision] Aligning spans across {len(records_to_process)} notes...")
        for idx, row in tqdm(records_to_process.iterrows(), total=len(records_to_process)):
            note_id = row["ID"]
            note_text = row["Note"]
            target_json = json.loads(row["json"]) if isinstance(row["json"], str) else row["json"]

            spans_res = self.find_spans_for_note(note_text, target_json)

            total_target_syms += spans_res["target_symptoms_count"]
            total_aligned_syms += spans_res["aligned_symptoms_count"]
            total_negated_found += len(spans_res["negated_spans"])
            total_ambiguous_found += len(spans_res["ambiguous_spans"])

            record_entry = {
                "id": note_id,
                "text": note_text,
                "aligned_spans": spans_res["aligned_spans"],
                "ambiguous_spans": spans_res["ambiguous_spans"],
                "negated_spans": spans_res["negated_spans"]
            }
            dataset_annotations.append(record_entry)

            # Keep 20 formatted samples for manual review template
            if len(manual_review_samples) < 20:
                manual_review_samples.append({
                    "id": note_id,
                    "note_snippet": note_text[:400],
                    "candidate_spans": spans_res["aligned_spans"],
                    "negated_mentions": spans_res["negated_spans"],
                    "ambiguous_mentions": spans_res["ambiguous_spans"],
                    "ground_truth_json": target_json,
                    "human_annotator_verification": {
                        "verified_by": "UNVERIFIED_WEAK_LABEL",
                        "notes_clean": True,
                        "approved_for_gold_benchmark": False
                    }
                })

        overall_coverage = total_aligned_syms / total_target_syms if total_target_syms > 0 else 0.0

        provenance_data = {
            "total_notes_aligned": len(records_to_process),
            "total_target_symptoms": total_target_syms,
            "total_aligned_symptoms": total_aligned_syms,
            "overall_symptom_span_alignment_rate": round(overall_coverage, 4),
            "total_negated_spans_filtered": total_negated_found,
            "total_ambiguous_spans_isolated": total_ambiguous_found,
            "provenance_note": (
                "Weak supervision uses distant supervision matching target JSON fields "
                "to textual spans. Negated mentions are filtered by the context assertion layer. "
                "Tokens in ambiguous spans are masked with -100 during token classification training "
                "to prevent false negative supervision."
            )
        }

        os.makedirs(os.path.dirname(output_provenance_path), exist_ok=True)
        with open(output_provenance_path, "w") as f:
            json.dump(provenance_data, f, indent=2)

        with open(output_gold_template_path, "w") as f:
            json.dump(manual_review_samples, f, indent=2)

        print("\n" + "=" * 60)
        print("WEAK SUPERVISION ALIGNMENT SUMMARY")
        print("=" * 60)
        print(f"Notes Processed            : {len(records_to_process)}")
        print(f"Symptom Span Alignment Rate: {overall_coverage*100:.2f}% ({total_aligned_syms}/{total_target_syms})")
        print(f"Negated Spans Filtered     : {total_negated_found}")
        print(f"Ambiguous Spans Isolated   : {total_ambiguous_found}")
        print(f"Provenance Saved To        : {output_provenance_path}")
        print(f"Manual Gold Template Saved : {output_gold_template_path}")
        print("=" * 60)

        return dataset_annotations


if __name__ == "__main__":
    aligner = WeakSupervisionAligner()
    train_df = pd.read_csv("train (1).csv")
    aligner.generate_weakly_supervised_dataset(train_df.head(500))
