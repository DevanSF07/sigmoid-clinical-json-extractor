"""
Outlines FSM Constrained Decoding & Pydantic Schema Validator
Integrates Outlines with the fine-tuned model and strict Pydantic PatientRecord schema.
Implements:
- FSM logit masking during autoregressive decode
- Pydantic v2 validation
- Bounded retries with failure logging
- Explicit tracking of raw generation validity vs retry success
- Strict prohibition of data fabrication
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, List
from pydantic import ValidationError

from schemas.patient_schema import (
    PatientRecord,
    PatientInfo,
    VitalSigns,
    ValueUnit,
    BloodPressure,
    BPComponent
)


class OutlinesSchemaGenerator:
    def __init__(
        self,
        transformers_model=None,
        tokenizer=None,
        max_retries: int = 2
    ):
        """
        Initializes Outlines FSM constrained generator.
        Wraps transformers model with Outlines SteerableGenerator using PatientRecord schema.
        """
        self.model = transformers_model
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.generator = None

        if self.model is not None and self.tokenizer is not None:
            self._init_outlines_generator()

    def _init_outlines_generator(self):
        """Build Outlines SteerableGenerator constrained by PatientRecord schema."""
        try:
            import outlines
            print("[Outlines] Wrapping Transformers model with Outlines backend...")
            outlines_model = outlines.from_transformers(self.model, self.tokenizer)
            self.generator = outlines.Generator(outlines_model, output_type=PatientRecord)
            print("[Outlines] Successfully initialized FSM constrained generator for PatientRecord.")
        except Exception as e:
            print(f"[Outlines Warning] Could not initialize Outlines SteerableGenerator: {e}")
            self.generator = None

    def generate_record(
        self,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Generate structured record via FSM-constrained decoding.
        Tracks:
        - raw_generation_valid (True/False)
        - retries_used (int)
        - record (PatientRecord or None)
        - error (str or None)
        Never fabricates records to force validation.
        """
        if self.generator is None:
            raise RuntimeError(
                "Outlines generator is not initialized with an active model. "
                "Ensure a fine-tuned or local causal LM is loaded."
            )

        attempts = 0
        raw_valid = False

        while attempts <= self.max_retries:
            attempts += 1
            try:
                # Outlines constrained generation
                response = self.generator(prompt)

                if isinstance(response, PatientRecord):
                    if attempts == 1:
                        raw_valid = True
                    return {
                        "success": True,
                        "raw_valid": raw_valid,
                        "retries_used": attempts - 1,
                        "record": response,
                        "error": None
                    }
                elif isinstance(response, str):
                    # Validate string output via Pydantic
                    parsed = json.loads(response)
                    record = PatientRecord(**parsed)
                    if attempts == 1:
                        raw_valid = True
                    return {
                        "success": True,
                        "raw_valid": raw_valid,
                        "retries_used": attempts - 1,
                        "record": record,
                        "error": None
                    }

            except (ValidationError, json.JSONDecodeError, Exception) as e:
                err_msg = str(e)
                print(f"[Outlines] Generation attempt {attempts} failed validation: {err_msg}")
                if attempts > self.max_retries:
                    return {
                        "success": False,
                        "raw_valid": False,
                        "retries_used": attempts - 1,
                        "record": None,
                        "error": err_msg
                    }

        return {
            "success": False,
            "raw_valid": False,
            "retries_used": attempts - 1,
            "record": None,
            "error": "Exceeded maximum retry budget."
        }

    @staticmethod
    def validate_candidate_dict(raw_data: Dict[str, Any]) -> Tuple[bool, Optional[PatientRecord], Optional[str]]:
        """
        Validate an extracted dictionary against the strict PatientRecord Pydantic schema.
        Preserves None values without inserting fake defaults (no default 45 or Male).
        """
        try:
            # Reconstruct patient_info
            p_info = raw_data.get("patient_info", {})
            age = p_info.get("age")
            if age is not None:
                try:
                    age = int(age)
                except (ValueError, TypeError):
                    age = None

            gender = p_info.get("gender")
            if gender is not None:
                gender = str(gender).strip().capitalize()
                if gender not in ["Male", "Female"]:
                    gender = "Male" if gender.lower().startswith("m") else "Female" if gender.lower().startswith("f") else None

            patient_info_obj = PatientInfo(age=age, gender=gender)

            # Reconstruct visit_motivation
            motivation = raw_data.get("visit_motivation")
            if motivation:
                motivation = str(motivation).strip()
            else:
                motivation = None

            # Reconstruct symptoms
            symptoms = list(raw_data.get("symptoms", []))

            # Reconstruct vital signs
            vitals = raw_data.get("vital_signs", {})
            vital_kwargs = {}

            if "blood_pressure" in vitals and isinstance(vitals["blood_pressure"], dict):
                bp = vitals["blood_pressure"]
                sys_v = bp.get("systolic", {}).get("value")
                dia_v = bp.get("diastolic", {}).get("value")
                if sys_v is not None and dia_v is not None:
                    vital_kwargs["blood_pressure"] = BloodPressure(
                        systolic=BPComponent(value=int(sys_v)),
                        diastolic=BPComponent(value=int(dia_v))
                    )

            for field_name in [
                "temperature",
                "heart_rate",
                "respiratory_rate",
                "oxygen_saturation",
                "cholesterol_level",
                "glucose_level"
            ]:
                if field_name in vitals and isinstance(vitals[field_name], dict):
                    v_entry = vitals[field_name]
                    val = v_entry.get("value")
                    unit = v_entry.get("unit")
                    if val is not None and unit is not None:
                        vital_kwargs[field_name] = ValueUnit(value=float(val), unit=str(unit))

            vitals_obj = VitalSigns(**vital_kwargs)

            record = PatientRecord(
                patient_info=patient_info_obj,
                visit_motivation=motivation,
                symptoms=symptoms,
                vital_signs=vitals_obj
            )

            return True, record, None

        except (ValidationError, Exception) as e:
            return False, None, str(e)
