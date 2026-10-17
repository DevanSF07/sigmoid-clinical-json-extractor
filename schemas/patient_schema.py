"""
Patient Record Pydantic Schemas (Placement Interview Edition)
Strict structured output schema for clinical entity extraction.
Enforces typed nested models, explicit nullability (no hallucinated defaults),
unit validation, and submission compatibility.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator


# Canonical allowed categories supported by the dataset
ALLOWED_VISIT_MOTIVATIONS = [
    "Allergies",
    "Anemia",
    "Anxiety Disorders",
    "Asthma",
    "COVID-19",
    "Chronic Obstructive Pulmonary Disease (COPD)",
    "Common Cold",
    "Depression",
    "Diabetes (Type 2)",
    "Ear Infection (Otitis Media)",
    "Eczema (Atopic Dermatitis)",
    "Gastroesophageal Reflux Disease (GERD)",
    "Heart Disease (Coronary Artery Disease)",
    "Hypertension (High Blood Pressure)",
    "Influenza (Flu)",
    "Pneumonia",
    "Sinusitis",
    "Strep Throat",
    "Tuberculosis (TB)",
    "Urinary Tract Infection (UTI)"
]

ALLOWED_SYMPTOMS = [
    "abdominal_pain",
    "anxiety",
    "blurred_vision",
    "chest_pain",
    "cough",
    "diarrhea",
    "difficulty_breathing",
    "difficulty_concentrating",
    "dizziness",
    "dry_skin",
    "ear_pain",
    "facial_pain",
    "fatigue",
    "fever",
    "frequent_urination",
    "headache",
    "heartburn",
    "increased_thirst",
    "itchy_eyes",
    "joint_pain",
    "loss_of_taste_smell",
    "nausea",
    "night_sweats",
    "painful_urination",
    "pale_skin",
    "rash",
    "restlessness",
    "runny_nose",
    "sadness",
    "sneezing",
    "sore_throat",
    "swollen_lymph_nodes",
    "vomiting",
    "weight_loss",
    "wheezing"
]

ALLOWED_UNITS = {
    "temperature": ["°C", "deg C", "C"],
    "heart_rate": ["bpm"],
    "respiratory_rate": ["breaths/min"],
    "oxygen_saturation": ["%"],
    "glucose_level": ["mg/dL"],
    "cholesterol_level": ["mg/dL"],
    "blood_pressure": ["mmHg"]
}


class ValueUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    unit: str

    @field_validator("value", mode="before")
    @classmethod
    def round_value(cls, v):
        if v is None:
            raise ValueError("Vital sign value cannot be None")
        return round(float(v), 2)

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, u: str) -> str:
        clean = u.strip()
        # Canonicalize degree symbol
        if clean in ["°C", "deg C", "C", "degC"]:
            return "°C"
        return clean


class BPComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int
    unit: str = "mmHg"

    @field_validator("unit")
    @classmethod
    def validate_bp_unit(cls, u: str) -> str:
        if u.strip() != "mmHg":
            raise ValueError(f"Invalid blood pressure unit '{u}'. Expected 'mmHg'.")
        return "mmHg"


class BloodPressure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    systolic: BPComponent
    diastolic: BPComponent


class PatientInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Strictly nullable: never invent default age or gender
    age: Optional[int] = Field(None, ge=0, le=130, description="Patient age in years; None if not stated.")
    gender: Optional[str] = Field(None, description="Patient gender (Male or Female); None if not stated.")

    @field_validator("gender")
    @classmethod
    def standardize_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = str(v).strip().capitalize()
        if clean in ["Male", "Female"]:
            return clean
        if clean.lower().startswith("m"):
            return "Male"
        if clean.lower().startswith("f"):
            return "Female"
        return clean


class VitalSigns(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: Optional[ValueUnit] = None
    heart_rate: Optional[ValueUnit] = None
    respiratory_rate: Optional[ValueUnit] = None
    oxygen_saturation: Optional[ValueUnit] = None
    cholesterol_level: Optional[ValueUnit] = None
    glucose_level: Optional[ValueUnit] = None
    blood_pressure: Optional[BloodPressure] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize populated vital signs matching the dataset JSON format."""
        result = {}
        if self.blood_pressure is not None:
            result["blood_pressure"] = {
                "systolic": {
                    "value": self.blood_pressure.systolic.value,
                    "unit": self.blood_pressure.systolic.unit
                },
                "diastolic": {
                    "value": self.blood_pressure.diastolic.value,
                    "unit": self.blood_pressure.diastolic.unit
                }
            }
        for field_name in [
            "temperature",
            "heart_rate",
            "respiratory_rate",
            "oxygen_saturation",
            "cholesterol_level",
            "glucose_level"
        ]:
            val = getattr(self, field_name)
            if val is not None:
                result[field_name] = {
                    "value": val.value,
                    "unit": val.unit
                }
        return result


class PatientRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_info: PatientInfo
    # Strictly nullable: never default to 'Common Cold'
    visit_motivation: Optional[str] = Field(
        None,
        description="Primary clinical visit reason from canonical taxonomy; None if not present."
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Extracted clinical symptoms list adhering to canonical vocabulary."
    )
    vital_signs: VitalSigns = Field(
        default_factory=VitalSigns,
        description="Structured patient vital signs with validated values and units."
    )
    # Medication note: Ground truth JSON in this dataset contains 0 medication annotations.
    # We support the field in the schema as nullable without hallucinating labels.
    medications: Optional[List[str]] = Field(
        None,
        description="Optional extracted medications. Nullable because dataset target JSON lacks medication labels."
    )

    @field_validator("visit_motivation")
    @classmethod
    def validate_motivation(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if clean in ALLOWED_VISIT_MOTIVATIONS:
            return clean
        # Match case-insensitively
        for canonical in ALLOWED_VISIT_MOTIVATIONS:
            if clean.lower() == canonical.lower():
                return canonical
        return clean

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, syms: List[str]) -> List[str]:
        clean_syms = set()
        for s in syms:
            s_clean = s.strip().lower()
            if s_clean in ALLOWED_SYMPTOMS:
                clean_syms.add(s_clean)
        return sorted(list(clean_syms))

    def to_clean_dict(self) -> Dict[str, Any]:
        """Serialize to standard JSON-compatible dictionary."""
        data = {
            "patient_info": {
                "age": self.patient_info.age,
                "gender": self.patient_info.gender
            },
            "visit_motivation": self.visit_motivation,
            "symptoms": sorted(list(set(self.symptoms))),
            "vital_signs": self.vital_signs.to_dict()
        }
        if self.medications is not None:
            data["medications"] = self.medications
        return data

    def to_submission_dict(self) -> Dict[str, Any]:
        """Compatible format with existing submission CSV and ground truth."""
        return {
            "patient_info": {
                "age": self.patient_info.age,
                "gender": self.patient_info.gender
            },
            "visit_motivation": self.visit_motivation or "",
            "symptoms": sorted(list(set(self.symptoms))),
            "vital_signs": self.vital_signs.to_dict()
        }
