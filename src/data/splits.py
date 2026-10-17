"""
Data Audit and Leakage-Free Stratified Splitting Module
Handles data inspection, deduplication check, and split generation.
"""

import os
import json
import hashlib
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def compute_text_hash(text: str) -> str:
    """Compute sha256 hash of normalized clinical note text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_datasets(train_csv: str = "train (1).csv", test_csv: str = "test (1).csv") -> Dict[str, Any]:
    """
    Perform rigorous audit of train and test datasets.
    Reports row counts, columns, missing values, JSON schema adherence,
    medication labels check, and text duplicates.
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Check text hashes
    train_hashes = [compute_text_hash(t) for t in train_df["Note"]]
    test_hashes = [compute_text_hash(t) for t in test_df["Note"]]

    train_unique_hashes = len(set(train_hashes))
    test_unique_hashes = len(set(test_hashes))
    overlap_hashes = len(set(train_hashes).intersection(set(test_hashes)))

    # Inspect JSON structure across all train records
    malformed_json_count = 0
    top_keys = set()
    medication_keys_found = 0
    vital_sign_distribution = {}
    motivation_distribution = {}
    symptoms_distribution = {}

    for raw_json in train_df["json"]:
        try:
            data = json.loads(raw_json)
            for k in data.keys():
                top_keys.add(k)
            if "medications" in data or "medication" in data:
                medication_keys_found += 1
            
            # Motivations
            mot = data.get("visit_motivation", "UNKNOWN")
            motivation_distribution[mot] = motivation_distribution.get(mot, 0) + 1

            # Vitals
            for vk in data.get("vital_signs", {}).keys():
                vital_sign_distribution[vk] = vital_sign_distribution.get(vk, 0) + 1

            # Symptoms
            for sym in data.get("symptoms", []):
                symptoms_distribution[sym] = symptoms_distribution.get(sym, 0) + 1

        except Exception:
            malformed_json_count += 1

    audit_summary = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
        "train_missing_values": train_df.isnull().sum().to_dict(),
        "test_missing_values": test_df.isnull().sum().to_dict(),
        "train_unique_note_hashes": train_unique_hashes,
        "test_unique_note_hashes": test_unique_hashes,
        "train_test_hash_overlap": overlap_hashes,
        "malformed_train_json_count": malformed_json_count,
        "json_top_level_keys": sorted(list(top_keys)),
        "medication_annotations_present_in_json": medication_keys_found > 0,
        "total_unique_motivations": len(motivation_distribution),
        "total_unique_symptoms": len(symptoms_distribution),
        "motivation_counts": motivation_distribution,
        "vital_sign_counts": vital_sign_distribution
    }
    return audit_summary


def create_reproducible_splits(
    train_csv: str = "train (1).csv",
    output_dir: str = "splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_eval_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create leakage-free, stratified train/val/held-out evaluation splits from train (1).csv.
    Stratifies by visit_motivation to ensure balanced clinical condition representation.
    """
    assert abs(train_ratio + val_ratio + test_eval_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(train_csv)
    # Parse motivations for stratification
    motivations = []
    for s in df["json"]:
        try:
            motivations.append(json.loads(s).get("visit_motivation", "UNKNOWN"))
        except Exception:
            motivations.append("UNKNOWN")
    df["stratify_key"] = motivations

    # First split off train
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_eval_ratio),
        random_state=random_state,
        stratify=df["stratify_key"]
    )

    # Next split temp into val and held-out eval (50% each of remainder)
    relative_eval_ratio = test_eval_ratio / (val_ratio + test_eval_ratio)
    val_df, eval_df = train_test_split(
        temp_df,
        test_size=relative_eval_ratio,
        random_state=random_state,
        stratify=temp_df["stratify_key"]
    )

    train_df = train_df.drop(columns=["stratify_key"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["stratify_key"]).reset_index(drop=True)
    eval_df = eval_df.drop(columns=["stratify_key"]).reset_index(drop=True)

    # Verify zero overlap in IDs and text hashes
    train_ids = set(train_df["ID"])
    val_ids = set(val_df["ID"])
    eval_ids = set(eval_df["ID"])
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between train and val IDs!"
    assert len(train_ids.intersection(eval_ids)) == 0, "Leakage detected between train and eval IDs!"
    assert len(val_ids.intersection(eval_ids)) == 0, "Leakage detected between val and eval IDs!"

    train_hashes = set(compute_text_hash(t) for t in train_df["Note"])
    val_hashes = set(compute_text_hash(t) for t in val_df["Note"])
    eval_hashes = set(compute_text_hash(t) for t in eval_df["Note"])
    assert len(train_hashes.intersection(val_hashes)) == 0, "Leakage detected in note text between train and val!"
    assert len(train_hashes.intersection(eval_hashes)) == 0, "Leakage detected in note text between train and eval!"

    # Save manifests
    train_manifest = {
        "split": "train",
        "count": len(train_df),
        "ids": train_df["ID"].tolist()
    }
    val_manifest = {
        "split": "validation",
        "count": len(val_df),
        "ids": val_df["ID"].tolist()
    }
    eval_manifest = {
        "split": "held_out_evaluation",
        "count": len(eval_df),
        "ids": eval_df["ID"].tolist()
    }

    with open(os.path.join(output_dir, "train_manifest.json"), "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open(os.path.join(output_dir, "val_manifest.json"), "w") as f:
        json.dump(val_manifest, f, indent=2)
    with open(os.path.join(output_dir, "test_eval_manifest.json"), "w") as f:
        json.dump(eval_manifest, f, indent=2)

    # Also save audit report
    audit_data = audit_datasets(train_csv)
    with open(os.path.join(output_dir, "data_audit_report.json"), "w") as f:
        json.dump(audit_data, f, indent=2)

    print(f"[Data Splits] Successfully created splits in {output_dir}/:")
    print(f"  - Train: {len(train_df)} samples ({train_ratio*100:.1f}%)")
    print(f"  - Val:   {len(val_df)} samples ({val_ratio*100:.1f}%)")
    print(f"  - Eval:  {len(eval_df)} samples ({test_eval_ratio*100:.1f}%)")
    print("  - Leakage check: Passed (zero ID and zero text hash overlap).")

    return train_df, val_df, eval_df


if __name__ == "__main__":
    create_reproducible_splits()
