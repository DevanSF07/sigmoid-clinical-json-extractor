"""Generate group-disjoint NER predictions for LLM training using K-fold training.

Each fold model trains on the other groups, using the independent validation set
for checkpoint selection. Expensive by design; no gold entities enter LLM prompts.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from clinical.data import near_signature
from clinical.io import read_records,write_jsonl

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',default='artifacts/oof');p.add_argument('--folds',type=int,default=5);p.add_argument('--epochs',type=int,default=5)
    a=p.parse_args()
    if a.folds<2:raise ValueError('Need at least two folds')
    root=Path(a.output);rows=read_records(a.train);result=[]
    fold=lambda r:int(hashlib.sha256(near_signature(r['note']).encode()).hexdigest()[:8],16)%a.folds
    for k in range(a.folds):
        train=[r for r in rows if fold(r)!=k];test=[r for r in rows if fold(r)==k]
        if not train or not test:raise ValueError('Empty fold; reduce folds or add examples')
        train_path=root/f'fold{k}_train.jsonl';test_path=root/f'fold{k}_heldout.jsonl'
        pred_path=root/f'fold{k}_predictions.jsonl';checkpoint=root/f'fold{k}_ner'
        write_jsonl(train_path,train);write_jsonl(test_path,test)
        subprocess.run([sys.executable,'-m','models.train_ner','--train',str(train_path),'--validation',a.validation,
            '--output',str(checkpoint),'--epochs',str(a.epochs)],check=True)
        subprocess.run([sys.executable,'-m','scripts.predict_entities','--input',str(test_path),
            '--checkpoint',str(checkpoint),'--output',str(pred_path)],check=True)
        for r in read_records(pred_path):result.append({**r,'out_of_fold':True,'fold':k})
    write_jsonl(root/'train_entities.jsonl',result)
if __name__=='__main__':main()
