import csv
import hashlib
import json
import re
from pathlib import Path

def fingerprint(note):
    return hashlib.sha256(re.sub(r"\s+", " ", note.lower()).strip().encode()).hexdigest()

def read_records(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            item = {'id': str(row['ID']), 'note': row['Note']}
            if 'json' in row:
                item['target'] = json.loads(row['json'])
            records.append(item)
    return records

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
