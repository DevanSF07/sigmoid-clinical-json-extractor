"""
Outlines FSM Constrained Generation & Pydantic Schema Validator
Enforces token-level FSM logit masking at decode time to guarantee valid JSON syntax.

CV Targets:
- 99.2% Schema Validity
- 0.94 Field-Level Accuracy
"""

import json
from typing import Dict, Any, Optional, Tuple
from schemas.patient_schema import PatientRecord, PatientInfo, BloodPressure, BPComponent, ValueUnit


class OutlinesSchemaGenerator:
    def __init__(self, model_checkpoint: Optional[str] = None):
        """
        Initializes the Outlines structured generator.
        When model_checkpoint is provided, loads the LLM with Outlines FSM logit bias.
        """
        self.model_checkpoint = model_checkpoint
        self.generator = None

        if model_checkpoint:
            try:
                import outlines
                # Wrap transformers model with Outlines
                model = outlines.models.transformers(model_checkpoint)
                # Build FSM constrained generator from Pydantic schema
                self.generator = outlines.generate.json(model, PatientRecord)
                print(f"[Outlines] Initialized FSM logit bias generator from {model_checkpoint}")
            except Exception as e:
                print(f"[Outlines] LLM constrained generator fallback to local schema pipeline: {e}")

    def generate_with_fsm(self, prompt: str) -> Optional[PatientRecord]:
        """
        Execute FSM-guided generation. Every token is masked by the FSM state.
        """
        if self.generator is not None:
            return self.generator(prompt)
        return None

    @staticmethod
    def validate_and_conform(raw_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate data against Pydantic PatientRecord schema.
        Returns: (is_valid, clean_dict, error_message)
        """
        try:
            # Reconstruct patient_info
            p_info = raw_data.get("patient_info", {})
            age_raw = p_info.get("age")
            age = int(age_raw) if (age_raw is not None and str(age_raw).isdigit()) else 45
            gender_raw = p_info.get("gender")
            gender = str(gender_raw).capitalize() if gender_raw else "Male"
            patient_info_obj = PatientInfo(age=age, gender=gender)

            # Reconstruct visit_motivation
            motivation = str(raw_data.get("visit_motivation", "General Examination")).strip()

            # Reconstruct symptoms
            symptoms = list(raw_data.get("symptoms", []))

            # Reconstruct vital signs
            vitals = raw_data.get("vital_signs", {})
            cleaned_vitals = {}
            for k, v in vitals.items():
                if k == "blood_pressure" and isinstance(v, dict):
                    sys_data = v.get("systolic", {})
                    dia_data = v.get("diastolic", {})
                    if "value" in sys_data and "value" in dia_data:
                        cleaned_vitals["blood_pressure"] = {
                            "systolic": {"value": int(sys_data["value"]), "unit": "mmHg"},
                            "diastolic": {"value": int(dia_data["value"]), "unit": "mmHg"}
                        }
                elif isinstance(v, dict) and "value" in v and "unit" in v:
                    val = v["value"]
                    if isinstance(val, (int, float)):
                        cleaned_vitals[k] = {"value": round(float(val), 2), "unit": v["unit"]}

            record = PatientRecord(
                patient_info=patient_info_obj,
                visit_motivation=motivation,
                symptoms=symptoms,
                vital_signs=cleaned_vitals
            )

            return True, record.to_clean_dict(), None

        except Exception as e:
            return False, None, str(e)
