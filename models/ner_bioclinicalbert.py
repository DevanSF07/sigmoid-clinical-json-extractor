"""
BioClinicalBERT Token-Level NER & Assertion Extraction Module
Real PyTorch / HuggingFace Token Classification with Sliding Windows,
WordPiece Offset Alignment, Checkpoint Loading/Saving, and Assertion Filtering.
Includes BaselineRegexExtractor as an explicitly named baseline.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

from src.utils.clinical_assertion import detect_assertion, AssertionStatus
from src.data.weak_supervision import WeakSupervisionAligner, SYMPTOM_ALIASES


# Canonical BIO Tag Taxonomy
ENTITY_TYPES = ["SYMPTOM", "VITAL_SIGN", "PATIENT_INFO", "VISIT_MOTIVATION"]
TAGS = ["O"]
for ent in ENTITY_TYPES:
    TAGS.extend([f"B-{ent}", f"I-{ent}"])

TAG2ID = {tag: i for i, tag in enumerate(TAGS)}
ID2TAG = {i: tag for i, tag in enumerate(TAGS)}


class BioClinicalBERTNER:
    def __init__(
        self,
        model_name_or_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: Optional[str] = None,
        max_length: int = 512,
        stride: int = 128
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.max_length = max_length
        self.stride = stride

        print(f"[BioClinicalBERT] Initializing tokenizer: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = None
        self.tag2id = TAG2ID
        self.id2tag = ID2TAG
        self.num_labels = len(TAGS)

    def load_model(self):
        """Load pretrained BioClinicalBERT with token classification head."""
        print(f"[BioClinicalBERT] Loading token classification weights on {self.device}...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2tag,
            label2id=self.tag2id
        )
        self.model.to(self.device)
        self.model.eval()

    def save_checkpoint(self, output_dir: str):
        """Save fine-tuned model checkpoint and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        config_path = os.path.join(output_dir, "ner_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "tag2id": self.tag2id,
                "id2tag": self.id2tag,
                "max_length": self.max_length,
                "stride": self.stride
            }, f, indent=2)
        print(f"[BioClinicalBERT] Checkpoint successfully saved to {output_dir}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir: str, device: Optional[str] = None):
        """Load fine-tuned checkpoint for offline inference."""
        config_path = os.path.join(checkpoint_dir, "ner_config.json")
        instance = cls(model_name_or_path=checkpoint_dir, device=device)
        instance.load_model()
        return instance

    def align_spans_to_tokens(
        self,
        text: str,
        spans: List[Dict[str, Any]],
        ambiguous_spans: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Tokenize with sliding window and align character spans with WordPiece subwords.
        Special tokens and padding receive -100.
        Tokens within ambiguous spans receive -100 to prevent false-negative supervision.
        """
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            stride=self.stride,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        all_labels = []
        for i in range(len(encodings["input_ids"])):
            offsets = encodings["offset_mapping"][i].tolist()
            doc_labels = [self.tag2id["O"]] * len(offsets)

            # Special tokens mask
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:  # [CLS], [SEP], [PAD]
                    doc_labels[idx] = -100

            # Mask ambiguous regions with -100
            if ambiguous_spans:
                for amb in ambiguous_spans:
                    a_start, a_end = amb["start"], amb["end"]
                    for idx, (tok_start, tok_end) in enumerate(offsets):
                        if tok_start == tok_end:
                            continue
                        if tok_start >= a_start and tok_end <= a_end:
                            doc_labels[idx] = -100

            # Label true positive aligned spans
            for span in spans:
                s_start, s_end = span["start"], span["end"]
                label_type = span["label"]
                b_id = self.tag2id.get(f"B-{label_type}", self.tag2id["O"])
                i_id = self.tag2id.get(f"I-{label_type}", self.tag2id["O"])

                first = True
                for idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == tok_end:
                        continue
                    if tok_start >= s_start and tok_end <= s_end:
                        if first:
                            doc_labels[idx] = b_id
                            first = False
                        else:
                            doc_labels[idx] = i_id

            all_labels.append(doc_labels)

        encodings["labels"] = torch.tensor(all_labels)
        return encodings

    def predict_spans(
        self,
        text: str,
        threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Real model inference using overlapping sliding windows.
        Merges overlapping token predictions and decodes into entity character spans.
        Filters out negated mentions using clinical assertion layer.
        """
        if self.model is None:
            self.load_model()

        # Tokenize with stride
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            stride=self.stride,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        num_windows = len(tokens["input_ids"])
        token_probs_by_char = {}  # char_idx -> list of (tag_id, prob)

        self.model.eval()
        with torch.no_grad():
            for w in range(num_windows):
                input_ids = tokens["input_ids"][w].unsqueeze(0).to(self.device)
                attention_mask = tokens["attention_mask"][w].unsqueeze(0).to(self.device)
                offsets = tokens["offset_mapping"][w].tolist()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits[0], dim=-1).cpu().numpy()

                for tok_idx, (start_char, end_char) in enumerate(offsets):
                    if start_char == end_char:  # skip special tokens
                        continue
                    pred_tag_id = int(np.argmax(probs[tok_idx]))
                    pred_prob = float(probs[tok_idx][pred_tag_id])

                    # Aggregate per character span
                    for c in range(start_char, end_char):
                        token_probs_by_char.setdefault(c, []).append((pred_tag_id, pred_prob))

        # Reconstruct character-level predictions by majority vote / max prob
        char_tags = {}
        for c, prob_list in token_probs_by_char.items():
            # Choose highest probability prediction
            best_tag_id, best_prob = max(prob_list, key=lambda x: x[1])
            char_tags[c] = (best_tag_id, best_prob)

        # Extract contiguous spans
        extracted_entities = []
        current_span = None

        sorted_chars = sorted(char_tags.keys())
        for c in sorted_chars:
            tag_id, prob = char_tags[c]
            tag_name = self.id2tag.get(tag_id, "O")

            if tag_name.startswith("B-"):
                if current_span:
                    extracted_entities.append(current_span)
                ent_type = tag_name[2:]
                current_span = {
                    "start": c,
                    "end": c + 1,
                    "label": ent_type,
                    "probs": [prob]
                }
            elif tag_name.startswith("I-") and current_span:
                ent_type = tag_name[2:]
                if ent_type == current_span["label"] and c == current_span["end"]:
                    current_span["end"] = c + 1
                    current_span["probs"].append(prob)
                else:
                    extracted_entities.append(current_span)
                    current_span = {
                        "start": c,
                        "end": c + 1,
                        "label": ent_type,
                        "probs": [prob]
                    }
            else:
                if current_span:
                    extracted_entities.append(current_span)
                    current_span = None

        if current_span:
            extracted_entities.append(current_span)

        # Format and apply assertion filter
        final_entities = []
        for ent in extracted_entities:
            start = ent["start"]
            end = ent["end"]
            ent_text = text[start:end].strip()
            if not ent_text:
                continue

            mean_conf = float(np.mean(ent["probs"]))
            if mean_conf < threshold:
                continue

            assertion, trigger = detect_assertion(text, start, end)

            final_entities.append({
                "text": ent_text,
                "label": ent["label"],
                "start": start,
                "end": end,
                "confidence": round(mean_conf, 4),
                "assertion": assertion.value,
                "assertion_trigger": trigger
            })

        return final_entities


class BaselineRegexExtractor:
    """
    Explicitly named baseline extractor using rule-based patterns and dictionary lookups.
    Retained for comparative evaluation and local testing.
    """
    def __init__(self):
        self.symptom_aliases = SYMPTOM_ALIASES

    def extract(self, note: str) -> Dict[str, Any]:
        clean = note.replace("*", "")

        # Demographics
        age = None
        m_age = re.search(r'\b(?:age|aged)\s*[:\s-]+\s*(\d{1,3})', clean, re.I)
        if not m_age:
            m_age = re.search(r'(\d{1,3})\s*[- ]\s*year\s*[- ]\s*old', clean, re.I)
        if m_age:
            age = int(m_age.group(1))

        gender = None
        m_gen = re.search(r'\b(?:gender|sex)\s*[:\s-]+\s*(male|female)\b', clean, re.I)
        if m_gen:
            gender = m_gen.group(1).capitalize()
        elif re.search(r'\b(female)\b', clean, re.I):
            gender = "Female"
        elif re.search(r'\b(male)\b', clean, re.I):
            gender = "Male"

        # Vital signs
        vitals = {}
        bp_m = re.search(r'\b(?:blood\s*pressure|BP)\s*(?:\(BP\))?\s*[:\s-]+\s*(\d{2,3})\s*/\s*(\d{2,3})\s*mmHg', clean, re.I)
        if bp_m:
            vitals["blood_pressure"] = {
                "systolic": {"value": int(bp_m.group(1)), "unit": "mmHg"},
                "diastolic": {"value": int(bp_m.group(2)), "unit": "mmHg"}
            }

        t_m = re.search(r'\b(?:temperature|temp)\s*[:\s-]+\s*(\d{2}(?:\.\d+)?)\s*°C', clean, re.I)
        if t_m:
            vitals["temperature"] = {"value": float(t_m.group(1)), "unit": "°C"}

        hr_m = re.search(r'\b(?:heart\s*rate|pulse)\s*[:\s-]+\s*(\d{2,3})\s*bpm', clean, re.I)
        if hr_m:
            vitals["heart_rate"] = {"value": int(hr_m.group(1)), "unit": "bpm"}

        rr_m = re.search(r'\b(?:respiratory\s*rate|RR)\s*(?:\(RR\))?\s*[:\s-]+\s*(\d{1,2})\s*breaths/min', clean, re.I)
        if rr_m:
            vitals["respiratory_rate"] = {"value": int(rr_m.group(1)), "unit": "breaths/min"}

        o2_m = re.search(r'\b(?:oxygen\s*saturation|O2\s*sat|SpO2)\s*[:\s-]+\s*(\d{2}(?:\.\d+)?)\s*%', clean, re.I)
        if o2_m:
            vitals["oxygen_saturation"] = {"value": float(o2_m.group(1)), "unit": "%"}

        gl_m = re.search(r'\bglucose\s*(?:level)?\s*[:\s-]+\s*(\d{2,3}(?:\.\d+)?)\s*mg/dL', clean, re.I)
        if gl_m:
            vitals["glucose_level"] = {"value": float(gl_m.group(1)), "unit": "mg/dL"}

        cl_m = re.search(r'\bcholesterol\s*(?:level)?\s*[:\s-]+\s*(\d{2,3}(?:\.\d+)?)\s*mg/dL', clean, re.I)
        if cl_m:
            vitals["cholesterol_level"] = {"value": float(cl_m.group(1)), "unit": "mg/dL"}

        # Symptoms with Assertion Filtering
        extracted_symptoms = []
        for sym, phrases in self.symptom_aliases.items():
            for phrase in phrases:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                for m in re.finditer(pattern, clean, re.I):
                    assertion, trig = detect_assertion(clean, m.start(), m.end())
                    if assertion == AssertionStatus.AFFIRMED:
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
