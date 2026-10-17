"""
Patient Record Pydantic Schemas
Strict structured output schema for clinical entity extraction.
Used in conjunction with Outlines FSM constrained decoding.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class ValueUnit(BaseModel):
    value: Union[int, float]
    unit: str

    @field_validator("value")
    @classmethod
    def round_float(cls, v):
        if isinstance(v, float):
            return round(v, 2)
        return v


class BPComponent(BaseModel):
    value: int
    unit: str = "mmHg"


class BloodPressure(BaseModel):
    systolic: BPComponent
    diastolic: BPComponent


class PatientInfo(BaseModel):
    age: int = Field(..., ge=0, le=130, description="Patient age in years")
    gender: str = Field(..., description="Patient gender (Male or Female)")

    @field_validator("gender")
    @classmethod
    def standardize_gender(cls, v: str) -> str:
        clean = v.strip().capitalize()
        if clean in ["Male", "Female"]:
            return clean
        if clean.lower().startswith("m"):
            return "Male"
        return "Female"


class VitalSigns(BaseModel):
    temperature: Optional[ValueUnit] = None
    heart_rate: Optional[ValueUnit] = None
    respiratory_rate: Optional[ValueUnit] = None
    oxygen_saturation: Optional[ValueUnit] = None
    cholesterol_level: Optional[ValueUnit] = None
    glucose_level: Optional[ValueUnit] = None
    blood_pressure: Optional[BloodPressure] = None


class PatientRecord(BaseModel):
    patient_info: PatientInfo
    visit_motivation: str = Field(..., description="Primary clinical visit reason or chief complaint")
    symptoms: List[str] = Field(default_factory=list, description="Extracted clinical symptoms list")
    vital_signs: Dict[str, Any] = Field(default_factory=dict, description="Extracted patient vital signs")

    def to_clean_dict(self) -> Dict[str, Any]:
        """Serialize to standard JSON-compatible dictionary."""
        data = {
            "patient_info": {
                "age": self.patient_info.age,
                "gender": self.patient_info.gender
            },
            "visit_motivation": self.visit_motivation,
            "symptoms": sorted(list(set(self.symptoms))),
            "vital_signs": self.vital_signs
        }
        return data


# Canonical allowed categories
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
