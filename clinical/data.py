"""Audit and deterministic duplicate-grouped splitting without model dependencies."""
import argparse
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from clinical.io import read_records, write_jsonl, write_json, fingerprint
from schemas.patient_schema import PatientRecord

def near_signature(text):
    # Conservative template fingerprint: case, punctuation, whitespace and numeric
    # variations collapse. This does not detect all semantic paraphrases.
    text = re.sub(r'\d+(?:\.\d+)?', '#', text.lower())
    return re.sub(r'\W+', ' ', text).strip()

def split_records(records, seed=42):
    groups = defaultdict(list)
    for row in records:
        key = row.get('patient_id') or near_signature(row['note'])
        groups[str(key)].append(row)
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    splits = {'train': [], 'validation': [], 'evaluation': []}
    # Hash allocation is stable with respect to dataset ordering and keeps groups whole.
    for key in keys:
        u = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:8], 16) / 2**32
        split = 'train' if u < .8 else 'validation' if u < .9 else 'evaluation'
        splits[split].extend(groups[key])
    return splits

def prepare(input_path, output_dir, seed=42):
    rows = read_records(input_path)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Record IDs must be unique before splitting')
    invalid = []
    for r in rows:
        try:
            PatientRecord.model_validate(r['target'])
        except Exception as e:
            invalid.append({'id': r['id'], 'error': str(e)})
    if invalid:
        write_json(Path(output_dir)/'invalid_records.json', invalid)
        raise ValueError(f'{len(invalid)} invalid targets; see invalid_records.json')
    splits = split_records(rows, seed)
    for name, items in splits.items():
        write_jsonl(Path(output_dir)/f'{name}.jsonl', items)
    report = {
        'source_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        'seed': seed, 'rows': len(rows), 'invalid_targets': len(invalid),
        'exact_duplicate_rows': len(rows)-len({fingerprint(r['note']) for r in rows}),
        'template_groups': len({near_signature(r['note']) for r in rows}),
        'empty_notes': sum(not r['note'].strip() for r in rows),
        'categories': dict(Counter(r['target'].get('visit_motivation') for r in rows)),
        'target_keys': dict(Counter(k for r in rows for k in r['target'])),
        'splits': {k: len(v) for k, v in splits.items()},
        'manifest': {k: [r['id'] for r in v] for k, v in splits.items()},
        'supervision': 'Document JSON only; no gold character spans or medication targets.',
        'grouping': 'Explicit patient_id if supplied; otherwise normalized template fingerprint. CSV ID is a record ID, not a patient ID.',
        'limitation': 'Template grouping catches numeric/punctuation variants, not all semantic near duplicates. No patient IDs in source CSV.',
    }
    write_json(Path(output_dir)/'audit.json', report)
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input', default='train (1).csv'); p.add_argument('--output', default='artifacts/data'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); r=prepare(a.input,a.output,a.seed); print({k:v for k,v in r.items() if k!='manifest'})
