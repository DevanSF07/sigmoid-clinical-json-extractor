"""Typed, nullable records. Validation does not establish factual correctness."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from clinical.categories import ALLOWED_SYMPTOMS, ALLOWED_VISIT_MOTIVATIONS

Symptom = Literal[tuple(ALLOWED_SYMPTOMS)]
Motivation = Literal[tuple(ALLOWED_VISIT_MOTIVATIONS)]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class ValueUnit(StrictModel):
    value: float
    unit: str

class Temperature(ValueUnit):
    unit: Literal["°C", "°F"]

class HeartRate(ValueUnit):
    unit: Literal["bpm"]

class RespiratoryRate(ValueUnit):
    unit: Literal["breaths/min"]

class Saturation(ValueUnit):
    value: float = Field(ge=0, le=100)
    unit: Literal["%"]

class LabValue(ValueUnit):
    unit: Literal["mg/dL", "mmol/L"]

class BPComponent(StrictModel):
    value: StrictInt = Field(ge=0)
    unit: Literal["mmHg"] = "mmHg"

class BloodPressure(StrictModel):
    systolic: BPComponent
    diastolic: BPComponent

class PatientInfo(StrictModel):
    age: StrictInt | None = Field(default=None, ge=0, le=130)
    gender: Literal["Male", "Female", "Other", "Unknown"] | None = None

class VitalSigns(StrictModel):
    temperature: Temperature | None = None
    heart_rate: HeartRate | None = None
    respiratory_rate: RespiratoryRate | None = None
    oxygen_saturation: Saturation | None = None
    cholesterol_level: LabValue | None = None
    glucose_level: LabValue | None = None
    blood_pressure: BloodPressure | None = None

class PatientRecord(StrictModel):
    patient_info: PatientInfo
    visit_motivation: Motivation | None = None
    symptoms: list[Symptom] = Field(default_factory=list)
    vital_signs: VitalSigns = Field(default_factory=VitalSigns)

    def to_clean_dict(self):
        # Keep explicit unknown demographics, omit unobserved vital measurements.
        return {
            "patient_info": self.patient_info.model_dump(),
            "visit_motivation": self.visit_motivation,
            "symptoms": sorted(set(self.symptoms)),
            "vital_signs": self.vital_signs.model_dump(exclude_none=True),
        }
