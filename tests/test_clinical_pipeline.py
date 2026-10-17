"""
Unit and Integration Test Suite for Clinical Information Extraction
Covers:
- Nullable schema & extra field rejection
- Negation and clinical assertion detection
- Split leakage prevention
- Causal LM prompt & loss masking
- FAISS dense retrieval self-exclusion & serialization
- BioClinicalBERT sliding window token alignment
"""

import os
import json
import pytest
import torch
import pandas as pd
from transformers import AutoTokenizer

from schemas.patient_schema import (
    PatientRecord,
    PatientInfo,
    VitalSigns,
    ValueUnit,
    BloodPressure,
    BPComponent,
    ALLOWED_SYMPTOMS
)
from src.utils.clinical_assertion import detect_assertion, AssertionStatus
from src.data.splits import compute_text_hash, create_reproducible_splits
from models.ner_bioclinicalbert import BioClinicalBERTNER, BaselineRegexExtractor
from models.rag_retriever import ClinicalRAGRetriever
from models.train_llama_qlora import build_tokenized_inputs_and_labels


# =====================================================================
# 1. SCHEMA TESTS
# =====================================================================
def test_schema_nullability_no_defaults():
    """Verify missing demographics are strictly None, not defaulted to 45 or Male."""
    record = PatientRecord(
        patient_info=PatientInfo(age=None, gender=None),
        visit_motivation=None,
        symptoms=[],
        vital_signs=VitalSigns()
    )
    assert record.patient_info.age is None, "Missing age must remain None!"
    assert record.patient_info.gender is None, "Missing gender must remain None!"
    assert record.visit_motivation is None, "Missing motivation must remain None!"
    clean = record.to_clean_dict()
    assert clean["patient_info"]["age"] is None
    assert clean["patient_info"]["gender"] is None


def test_schema_extra_fields_forbidden():
    """Verify unexpected fields are rejected by Pydantic."""
    with pytest.raises(Exception):
        PatientInfo(age=30, gender="Male", fake_attribute="hallucinated")


def test_schema_blood_pressure_structure():
    """Verify blood pressure requires systolic and diastolic components."""
    bp = BloodPressure(
        systolic=BPComponent(value=120, unit="mmHg"),
        diastolic=BPComponent(value=80, unit="mmHg")
    )
    assert bp.systolic.value == 120
    assert bp.diastolic.value == 80


# =====================================================================
# 2. CLINICAL ASSERTION & NEGATION TESTS
# =====================================================================
def test_assertion_negation_detection():
    """Verify negated symptoms are flagged as NEGATED and excluded from active symptoms."""
    note = "Patient denies chest pain and fever, but complains of severe headache."
    
    # chest pain
    cp_start = note.index("chest pain")
    cp_status, trig = detect_assertion(note, cp_start, cp_start + len("chest pain"))
    assert cp_status == AssertionStatus.NEGATED
    assert trig == "denies"

    # headache (after 'but')
    ha_start = note.index("headache")
    ha_status, _ = detect_assertion(note, ha_start, ha_start + len("headache"))
    assert ha_status == AssertionStatus.AFFIRMED

    # Test baseline extractor exclusion
    base = BaselineRegexExtractor()
    extracted = base.extract(note)
    assert "chest_pain" not in extracted["symptoms"]
    assert "fever" not in extracted["symptoms"]
    assert "headache" in extracted["symptoms"]


def test_assertion_historical_and_uncertain():
    """Verify historical and uncertain mentions are identified."""
    note_hist = "Longstanding history of asthma since age 12."
    a_start = note_hist.index("asthma")
    hist_status, _ = detect_assertion(note_hist, a_start, a_start + len("asthma"))
    assert hist_status == AssertionStatus.HISTORICAL

    note_unc = "Chest X-ray shows possible pneumonia."
    p_start = note_unc.index("pneumonia")
    unc_status, _ = detect_assertion(note_unc, p_start, p_start + len("pneumonia"))
    assert unc_status == AssertionStatus.UNCERTAIN


# =====================================================================
# 3. SPLIT LEAKAGE TESTS
# =====================================================================
def test_split_leakage_prevention(tmp_path):
    """Verify zero overlap between train, val, and held-out evaluation splits."""
    dummy_data = pd.DataFrame({
        "ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Note": [f"Unique clinical note text case {i}." for i in range(10)],
        "json": [json.dumps({"visit_motivation": "Asthma" if i % 2 == 0 else "Allergies"}) for i in range(10)]
    })
    dummy_csv = os.path.join(tmp_path, "dummy_train.csv")
    dummy_data.to_csv(dummy_csv, index=False)

    train_df, val_df, eval_df = create_reproducible_splits(
        train_csv=dummy_csv,
        output_dir=str(tmp_path / "splits"),
        train_ratio=0.6,
        val_ratio=0.2,
        test_eval_ratio=0.2,
        random_state=42
    )

    t_ids = set(train_df["ID"])
    v_ids = set(val_df["ID"])
    e_ids = set(eval_df["ID"])

    assert len(t_ids.intersection(v_ids)) == 0
    assert len(t_ids.intersection(e_ids)) == 0
    assert len(v_ids.intersection(e_ids)) == 0


# =====================================================================
# 4. CAUSAL LM LOSS MASKING TESTS
# =====================================================================
def test_causal_lm_prompt_masking():
    """Verify -100 masks prompt and padding, leaving target JSON active."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    note = "Patient has acute bronchitis with cough."
    ner = {"symptoms": ["cough"]}
    target = '{"patient_info": {"age": null, "gender": null}, "symptoms": ["cough"]}'

    out = build_tokenized_inputs_and_labels(
        tokenizer=tokenizer,
        note=note,
        ner_entities=ner,
        target_json_str=target,
        max_seq_length=512
    )

    labels = out["labels"].tolist()
    assert labels[0] == -100, "First prompt token must be masked with -100!"
    active_tokens = [t for t in labels if t != -100]
    assert len(active_tokens) > 0, "Target tokens must be active for loss computation!"
    decoded = tokenizer.decode(active_tokens)
    assert '{"patient_info":' in decoded


# =====================================================================
# 5. RETRIEVAL TESTS
# =====================================================================
def test_retrieval_self_exclusion_and_metadata(tmp_path):
    """Verify query self-exclusion and index round-tripping."""
    retriever = ClinicalRAGRetriever(model_name="all-MiniLM-L6-v2")
    df = pd.DataFrame({
        "ID": [1, 2],
        "Note": [
            "Clinical presentation of pneumonia with fever.",
            "Clinical presentation of allergic rhinitis."
        ],
        "json": [
            '{"visit_motivation": "Pneumonia"}',
            '{"visit_motivation": "Allergies"}'
        ]
    })
    retriever.build_index(df, show_progress=False)

    # Query with query_id=1 should NOT retrieve note 1
    results = retriever.retrieve("Clinical presentation of pneumonia with fever.", query_id=1, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == 2, "Query note 1 was not excluded!"

    # Test saving & loading
    idx_p = str(tmp_path / "test.bin")
    meta_p = str(tmp_path / "test.json")
    retriever.save_index(idx_p, meta_p)

    retriever2 = ClinicalRAGRetriever(model_name="all-MiniLM-L6-v2")
    retriever2.load_index(idx_p, meta_p)
    assert retriever2.index.ntotal == 2


# =====================================================================
# 6. BIOCLINICALBERT WINDOWED ALIGNMENT TESTS
# =====================================================================
def test_bioclinicalbert_sliding_window_alignment():
    """Verify token alignment and special token masking."""
    bert = BioClinicalBERTNER()
    long_note = "Patient has cough. " * 80  # Long text
    spans = [{"start": 12, "end": 17, "label": "SYMPTOM"}]

    enc = bert.align_spans_to_tokens(long_note, spans)
    labels = enc["labels"][0].tolist()
    # Check that special tokens receive -100
    assert labels[0] == -100  # [CLS]
    # Check that symptom token received B-SYMPTOM
    assert bert.tag2id["B-SYMPTOM"] in labels
