"""Weak span supervision from JSON. Unmatched text is ignored, not gold O."""
import argparse
import ast
import re
from pathlib import Path
from clinical.assertions import assertion_status
from clinical.io import read_records, write_json, write_jsonl

# Reuse the baseline vocabulary without importing any model dependencies.
def symptom_aliases():
    tree = ast.parse((Path(__file__).parents[1]/'models/rules.py').read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='SYMPTOM_MAP' for t in n.targets):
            return ast.literal_eval(n.value)
    raise RuntimeError('Missing symptom vocabulary')

ALIASES = symptom_aliases()
ENTITY_TYPES = ['SYMPTOM', 'VITAL_SIGN', 'PATIENT_INFO']
TAGS = ['O'] + [tag for ent in ENTITY_TYPES for tag in (f'B-{ent}', f'I-{ent}')]


def weak_annotate(row):
    note, target = row['note'], row['target']
    entities, provenance, expected = [], [], 0
    def add_matches(pattern, label, canonical, assertion_filter=False):
        nonlocal expected
        expected += 1
        matches = list(re.finditer(pattern, note, re.I))
        usable = [m for m in matches if not assertion_filter or assertion_status(note,m.start(),m.end())=='present']
        for m in usable:
            entities.append({'start':m.start(),'end':m.end(),'label':label,'canonical':canonical})
        provenance.append({'canonical':canonical,'matches':len(usable),'source':'json_alias_alignment'})
    for symptom in target.get('symptoms',[]):
        phrases=ALIASES.get(symptom,[symptom.replace('_',' ')])
        add_matches(r'\b(?:'+'|'.join(re.escape(x) for x in sorted(phrases,key=len,reverse=True))+r')\b','SYMPTOM',symptom,True)
    info=target.get('patient_info',{})
    if info.get('age') is not None:
        # Require age context rather than matching an arbitrary identical vital value.
        age=re.escape(str(info['age']))
        add_matches(r'\b'+age+r'(?=\s*(?:[- ]year|years?\s*old))|(?<=age: )\b'+age+r'\b|(?<=aged )\b'+age+r'\b','PATIENT_INFO','age')
    if info.get('gender'):
        add_matches(r'\b'+re.escape(info['gender'])+r'\b','PATIENT_INFO','gender')
    # Align numeric values only inside a short field-specific vital expression.
    aliases={'temperature':r'(?:temperature|temp)','heart_rate':r'(?:heart rate|pulse)',
             'respiratory_rate':r'(?:respiratory rate|RR)','oxygen_saturation':r'(?:oxygen saturation|SpO2|O2 sat)',
             'cholesterol_level':r'cholesterol(?: level)?','glucose_level':r'glucose(?: level)?'}
    for key,value in target.get('vital_signs',{}).items():
        if key=='blood_pressure':
            s=value['systolic']['value']; d=value['diastolic']['value']
            add_matches(r'\b(?:BP|blood pressure)\s*(?:\(BP\))?\s*[: -]*'+str(s)+r'\s*/\s*'+str(d)+r'\s*mmHg','VITAL_SIGN',key)
        elif key in aliases:
            number=format(float(value['value']),'.10g')
            number_pattern=re.escape(number)+(r'(?:\.0+)?' if '.' not in number else '')
            unit=re.escape(value['unit'])
            add_matches(r'\b'+aliases[key]+r'\s*[: -]*'+number_pattern+r'\s*'+unit+r'(?!\w)','VITAL_SIGN',key)
    # Longest-first overlap resolution, then ascending offsets.
    selected=[]
    for ent in sorted(entities,key=lambda e: (-(e['end']-e['start']),e['start'])):
        if not any(ent['start']<e['end'] and e['start']<ent['end'] for e in selected):
            selected.append(ent)
    return {**row,'entities':sorted(selected,key=lambda e:e['start']),
            'annotation_source':'weak_json_alignment','complete_annotation':False,
            'alignment':provenance}


def align_token_labels(offsets, entities, complete=False, special_mask=None):
    """Works on original-document offsets, including overflow windows.

    BIO is restarted at a window boundary so a visible partial entity can be learned.
    """
    labels=[]; previous=None
    for i,(start,end) in enumerate(offsets):
        if start==end or (special_mask and special_mask[i]):
            labels.append(-100); previous=None; continue
        candidates=[(j,e) for j,e in enumerate(entities) if start<e['end'] and end>e['start']]
        if len(candidates)!=1:
            labels.append(0 if complete and not candidates else -100); previous=None; continue
        j,e=candidates[0]
        tag=('I-' if previous==j else 'B-')+e['label']
        labels.append(TAGS.index(tag)); previous=j
    return labels

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); rows=[weak_annotate(r) for r in read_records(a.input)]
    write_jsonl(a.output,rows)
    total=sum(len(r['alignment']) for r in rows); matched=sum(x['matches']>0 for r in rows for x in r['alignment'])
    write_json(str(a.output)+'.coverage.json',{'records':len(rows),'expected_fields':total,'matched_fields':matched,'coverage':matched/total if total else None,'source':'weak_json_alignment','gold_spans':False})
