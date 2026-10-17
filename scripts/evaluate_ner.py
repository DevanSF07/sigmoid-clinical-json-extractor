"""Evaluate real entity spans only against complete, reviewed gold documents."""
import argparse
from clinical.io import read_records,write_json
from clinical.ner_metrics import span_metrics, prf
from clinical.annotations import align_token_labels
from models.ner_bioclinicalbert import BioClinicalBERTNER

def main():
    p=argparse.ArgumentParser();p.add_argument('--gold',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='artifacts/ner_metrics.json')
    a=p.parse_args();rows=read_records(a.gold)
    if not rows or any(not r.get('complete_annotation') or r.get('annotation_source')!='manual_gold' for r in rows):
        raise ValueError('NER span F1 requires complete manual_gold annotations; weak alignment is not gold')
    model=BioClinicalBERTNER(a.checkpoint)
    predictions=[model.extract_entities(r['note']) for r in rows]
    result=span_metrics(predictions,[r['entities'] for r in rows])
    tp=fp=fn=0
    for row,pred in zip(rows,predictions):
        offsets=model.tokenizer(row['note'],return_offsets_mapping=True,truncation=False)['offset_mapping']
        truth=align_token_labels(offsets,row['entities'],complete=True)
        guess=align_token_labels(offsets,pred,complete=True)
        for t,g in zip(truth,guess):
            if t==-100:continue
            tp+=t!=0 and t==g
            fp+=g!=0 and t!=g
            fn+=t!=0 and t!=g
    write_json(a.output,{'entity_span_metrics':result,'token_bio_metrics':prf(tp,fp,fn),
        'records':len(rows),'source':'manual_gold'})
if __name__=='__main__':main()
