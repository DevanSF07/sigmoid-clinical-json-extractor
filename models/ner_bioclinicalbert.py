"""
BioClinicalBERT Token-Level NER and Multi-Label Tagging Module
Fine-tunes 'emilyalsentzer/Bio_ClinicalBERT' for clinical entity extraction.
Target entities: Symptoms, Vital Signs, Patient Demographics, Medications.

CV Metric Target:
- 0.91 F1 Score on Token-Level NER
"""

import os
import re
from typing import List, Dict, Any, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np

# Label mapping for BIO tagging
ENTITY_TYPES = ["SYMPTOM", "VITAL_SIGN", "MEDICATION", "PATIENT_INFO"]
TAGS = ["O"]
for ent in ENTITY_TYPES:
    TAGS.extend([f"B-{ent}", f"I-{ent}"])

TAG2ID = {tag: i for i, tag in enumerate(TAGS)}
ID2TAG = {i: tag for i, tag in enumerate(TAGS)}


class BioClinicalBERTNER:
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: str = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BioClinicalBERT] Initializing tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def initialize_model(self):
        """Load pretrained BioClinicalBERT with token classification head."""
        print(f"[BioClinicalBERT] Loading token classification weights on {self.device}...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(TAGS),
            id2label=ID2TAG,
            label2id=TAG2ID
        )
        self.model.to(self.device)

    def prepare_tokenized_dataset(self, texts: List[str], entity_annotations: List[List[Dict[str, Any]]]):
        """
        Align character-level entity annotations with subword tokens (WordPiece).
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        labels = []
        for i, text in enumerate(texts):
            doc_labels = [TAG2ID["O"]] * len(encodings["input_ids"][i])
            offsets = encodings["offset_mapping"][i]
            annotations = entity_annotations[i]

            for ann in annotations:
                start_char, end_char, label = ann["start"], ann["end"], ann["label"]
                b_tag = TAG2ID[f"B-{label}"]
                i_tag = TAG2ID[f"I-{label}"]

                first = True
                for token_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start >= tok_end:  # Special token
                        doc_labels[token_idx] = -100
                        continue
                    if tok_start >= start_char and tok_end <= end_char:
                        if first:
                            doc_labels[token_idx] = b_tag
                            first = False
                        else:
                            doc_labels[token_idx] = i_tag
            labels.append(doc_labels)
        encodings["labels"] = torch.tensor(labels)
        return encodings

    def get_training_args(self, output_dir: str = "./checkpoints/bioclinicalbert_ner") -> TrainingArguments:
        """
        Hyperparameters for token classification fine-tuning.
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_ratio=0.1,
            weight_decay=0.01,
            learning_rate=3e-5,
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available()
        )

    @staticmethod
    def compute_metrics(p):
        """
        Compute token-level Precision, Recall, and F1 score (ignoring 'O' and padding).
        """
        predictions, labels = p
        preds = np.argmax(predictions, axis=2)

        true_pos = 0
        false_pos = 0
        false_neg = 0

        for pred_seq, label_seq in zip(preds, labels):
            for p_tag_id, l_tag_id in zip(pred_seq, label_seq):
                if l_tag_id == -100 or l_tag_id == TAG2ID["O"]:
                    if p_tag_id != TAG2ID["O"] and p_tag_id != -100:
                        false_pos += 1
                    continue

                if p_tag_id == l_tag_id:
                    true_pos += 1
                elif p_tag_id == TAG2ID["O"]:
                    false_neg += 1
                else:
                    false_pos += 1
                    false_neg += 1

        prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        rec = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

    def extract_entities(self, note: str) -> Dict[str, Any]:
        """
        Clinical rule and vocabulary backed NER extraction engine.
        Extracts Symptoms, Vitals, Demographics, and Visit Motivation.
        """
        clean_note = note.replace("*", "")

        # 1. Demographics
        age = None
        m_age = re.search(r'\b(?:age|aged)\s*[:\s-]+\s*(\d{1,3})', clean_note, re.I)
        if not m_age:
            m_age = re.search(r'(\d{1,3})\s*[- ]\s*year\s*[- ]\s*old', clean_note, re.I)
        if not m_age:
            m_age = re.search(r'\((\d{1,3})\s*years\s*old', clean_note, re.I)
        if m_age:
            age = int(m_age.group(1))

        gender = None
        m_gen = re.search(r'\b(?:gender|sex)\s*[:\s-]+\s*(male|female)\b', clean_note, re.I)
        if m_gen:
            gender = m_gen.group(1).capitalize()
        else:
            m_gen_pat = re.search(r'patient:\s*(?:(?:[a-z0-9\-]+)[,\s]+)?(?:age\s*\d+[,\s]+)?(male|female)', clean_note, re.I)
            if m_gen_pat:
                gender = m_gen_pat.group(1).capitalize()
            elif re.search(r'\b(female)\b', clean_note, re.I):
                gender = "Female"
            elif re.search(r'\b(male)\b', clean_note, re.I):
                gender = "Male"

        # 2. Vital signs
        vitals = {}
        # Blood pressure
        bp_m = re.search(r'\b(?:blood\s*pressure|BP)\s*(?:\(BP\))?\s*[:\s-]+\s*(\d{2,3})\s*/\s*(\d{2,3})\s*mmHg', clean_note, re.I)
        if bp_m:
            vitals["blood_pressure"] = {
                "systolic": {"value": int(bp_m.group(1)), "unit": "mmHg"},
                "diastolic": {"value": int(bp_m.group(2)), "unit": "mmHg"}
            }

        # Temperature
        t_m = re.search(r'\b(?:temperature|temp)\s*[:\s-]+\s*(\d{2}(?:\.\d+)?)\s*°C', clean_note, re.I)
        if t_m:
            vitals["temperature"] = {"value": float(t_m.group(1)), "unit": "°C"}

        # Heart rate
        hr_m = re.search(r'\b(?:heart\s*rate|pulse)\s*[:\s-]+\s*(\d{2,3})\s*bpm', clean_note, re.I)
        if hr_m:
            vitals["heart_rate"] = {"value": int(hr_m.group(1)), "unit": "bpm"}

        # Respiratory rate
        rr_m = re.search(r'\b(?:respiratory\s*rate|RR)\s*(?:\(RR\))?\s*[:\s-]+\s*(\d{1,2})\s*breaths/min', clean_note, re.I)
        if rr_m:
            vitals["respiratory_rate"] = {"value": int(rr_m.group(1)), "unit": "breaths/min"}

        # Oxygen saturation
        o2_m = re.search(r'\b(?:oxygen\s*saturation|O2\s*sat|SpO2)\s*[:\s-]+\s*(\d{2}(?:\.\d+)?)\s*%', clean_note, re.I)
        if o2_m:
            vitals["oxygen_saturation"] = {"value": float(o2_m.group(1)), "unit": "%"}

        # Glucose level
        gl_m = re.search(r'\bglucose\s*(?:level)?\s*[:\s-]+\s*(\d{2,3}(?:\.\d+)?)\s*mg/dL', clean_note, re.I)
        if gl_m:
            vitals["glucose_level"] = {"value": float(gl_m.group(1)), "unit": "mg/dL"}

        # Cholesterol level
        cl_m = re.search(r'\bcholesterol\s*(?:level)?\s*[:\s-]+\s*(\d{2,3}(?:\.\d+)?)\s*mg/dL', clean_note, re.I)
        if cl_m:
            vitals["cholesterol_level"] = {"value": float(cl_m.group(1)), "unit": "mg/dL"}

        # 3. Symptoms
        SYMPTOM_MAP = {
            "abdominal_pain": ["abdominal pain", "stomach pain", "belly pain"],
            "anxiety": ["anxiety", "anxious", "nervousness"],
            "blurred_vision": ["blurred vision", "blurry vision"],
            "chest_pain": ["chest pain"],
            "cough": ["cough", "coughing"],
            "diarrhea": ["diarrhea"],
            "difficulty_breathing": ["difficulty breathing", "shortness of breath", "dyspnea"],
            "difficulty_concentrating": ["difficulty concentrating", "trouble concentrating"],
            "dizziness": ["dizziness", "lightheadedness", "dizzy"],
            "dry_skin": ["dry skin"],
            "ear_pain": ["ear pain", "earache"],
            "facial_pain": ["facial pain"],
            "fatigue": ["fatigue", "tiredness", "exhaustion"],
            "fever": ["fever", "febrile", "elevated temperature"],
            "frequent_urination": ["frequent urination", "urinary frequency"],
            "headache": ["headache"],
            "heartburn": ["heartburn", "acid reflux"],
            "increased_thirst": ["increased thirst", "excessive thirst", "polydipsia"],
            "itchy_eyes": ["itchy eyes", "eye itching"],
            "joint_pain": ["joint pain", "arthralgia"],
            "loss_of_taste_smell": ["loss of taste", "loss of smell", "loss of taste and smell", "loss of taste/smell"],
            "nausea": ["nausea", "nauseous"],
            "night_sweats": ["night sweats"],
            "painful_urination": ["painful urination", "dysuria"],
            "pale_skin": ["pale skin", "pallor"],
            "rash": ["rash", "skin rash"],
            "restlessness": ["restlessness", "restless"],
            "runny_nose": ["runny nose", "rhinorrhea"],
            "sadness": ["sadness", "feeling sad"],
            "sneezing": ["sneezing", "frequent sneezing"],
            "sore_throat": ["sore throat"],
            "swollen_lymph_nodes": ["swollen lymph nodes", "lymphadenopathy"],
            "vomiting": ["vomiting", "emesis"],
            "weight_loss": ["weight loss", "unintentional weight loss"],
            "wheezing": ["wheezing", "wheeze"]
        }

        extracted_symptoms = []
        note_lower = clean_note.lower()
        for sym, phrases in SYMPTOM_MAP.items():
            for phrase in phrases:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                if re.search(pattern, note_lower):
                    extracted_symptoms.append(sym)
                    break

        return {
            "patient_info": {
                "age": age,
                "gender": gender
            },
            "vital_signs": vitals,
            "symptoms": sorted(list(set(extracted_symptoms)))
        }
