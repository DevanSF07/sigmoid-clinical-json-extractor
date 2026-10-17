"""Explicit rule-based baseline; never substituted for a model."""
from typing import Dict, Any
import re
from clinical.assertions import assertion_status

class RuleExtractor:
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
                if any(assertion_status(clean_note, m.start(), m.end()) == "present" for m in re.finditer(pattern, note_lower)):
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
