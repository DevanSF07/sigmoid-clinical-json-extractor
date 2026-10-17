"""Field metrics with explicit denominators; failed records count as incorrect."""
import json
import math
from collections import Counter
from schemas.patient_schema import PatientRecord
from clinical.ner_metrics import prf

def flatten(obj,prefix=''):
    result={}
    if isinstance(obj,dict):
        for k,v in obj.items():result.update(flatten(v,f'{prefix}.{k}' if prefix else k))
    else:result[prefix]=obj
    return result

def equivalent(key,a,b,atol=.01):
    if a is None or b is None:return a==b
    if key.endswith('.value') and isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,abs_tol=atol,rel_tol=0)
    return a==b

def extraction_metrics(rows,predictions,atol=.01):
    by_id={str(p['id']):p for p in predictions}
    if len(by_id)!=len(predictions):raise ValueError('Duplicate prediction IDs')
    if set(by_id)-{str(r['id']) for r in rows}:raise ValueError('Predictions contain unknown IDs')
    counts=Counter();sym=Counter();fields=Counter();matched=Counter();n=len(rows)
    if not n:raise ValueError('No evaluation records')
    for row in rows:
        truth=PatientRecord.model_validate(row['target']).to_clean_dict()
        prediction=by_id.get(str(row['id']),{})
        raw=prediction.get('raw');pred=None
        try:
            parsed=json.loads(raw);counts['json_valid']+=1
        except (TypeError,ValueError):parsed=None
        if parsed is not None:
            try:pred=PatientRecord.model_validate(parsed).to_clean_dict();counts['schema_valid']+=1
            except Exception:pass
        if pred is None:counts['failed_records']+=1
        actual=pred or {}
        for key in ('age','gender'):
            counts[key+'_correct']+=pred is not None and actual['patient_info'][key]==truth['patient_info'][key]
        counts['motivation_correct']+=pred is not None and actual.get('visit_motivation')==truth.get('visit_motivation')
        t=set(truth['symptoms']);p=set(actual.get('symptoms',[]))
        sym['tp']+=len(t&p);sym['fp']+=len(p-t);sym['fn']+=len(t-p)
        t_flat=flatten(truth);p_flat=flatten(actual)
        # Symptoms are evaluated as a set; field score includes one exact set slot.
        t_flat['symptoms']=sorted(t);p_flat['symptoms']=sorted(p)
        keys=set(t_flat)|set(p_flat)
        for key in keys:
            fields['all']+=1
            good=pred is not None and key in t_flat and key in p_flat and equivalent(key,t_flat[key],p_flat[key],atol)
            matched['all']+=good
            if key.startswith('vital_signs.'):
                fields['vitals']+=1;matched['vitals']+=good
            if key in t_flat and t_flat[key] is not None:
                counts['supported_slots']+=1
                counts['missing_slots']+=key not in p_flat or p_flat.get(key) is None
            if key in p_flat and p_flat[key] is not None:
                counts['predicted_slots']+=1
                counts['unsupported_slots']+=key not in t_flat or t_flat.get(key) is None
        counts['exact_records']+=pred is not None and pred==truth
    ratio=lambda a,b:counts[a]/counts[b] if counts[b] else None
    return {'records':n,'failed_records':counts['failed_records'],
        'json_validity':counts['json_valid']/n,'schema_validity':counts['schema_valid']/n,
        'age_accuracy':counts['age_correct']/n,'gender_accuracy':counts['gender_correct']/n,
        'visit_motivation_accuracy':counts['motivation_correct']/n,'symptom_set':prf(**sym),
        'vital_leaf_accuracy':matched['vitals']/fields['vitals'] if fields['vitals'] else None,
        'field_leaf_accuracy':matched['all']/fields['all'] if fields['all'] else None,
        'exact_record_match':counts['exact_records']/n,
        'missing_slot_rate':ratio('missing_slots','supported_slots'),
        'unsupported_slot_rate':ratio('unsupported_slots','predicted_slots'),
        'numeric_absolute_tolerance':atol,
        'definition':'Union of gold/predicted leaf slots, symptoms one sorted-set slot. Missing prediction counts as failure. Unsupported means absent from gold, not clinician-verified hallucination. Units compared exactly; no implicit conversions.'}
