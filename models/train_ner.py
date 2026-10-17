"""Train BERT from annotated JSONL; weak examples keep unknown tokens masked."""
import argparse
from pathlib import Path
from clinical.annotations import TAGS, align_token_labels
from clinical.io import read_records, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',default='checkpoints/ner');p.add_argument('--base',default='emilyalsentzer/Bio_ClinicalBERT')
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, set_seed
    set_seed(a.seed);tokenizer=AutoTokenizer.from_pretrained(a.base,use_fast=True)
    train_rows=read_records(a.train);val_rows=read_records(a.validation)
    if any('entities' not in r for r in train_rows+val_rows):raise ValueError('Run weak annotation or provide reviewed entities first')
    def dataset(rows):
        examples=[]
        for row in rows:
            enc=tokenizer(row['note'],return_offsets_mapping=True,return_overflowing_tokens=True,
                          truncation=True,max_length=512,stride=96)
            for i,offsets in enumerate(enc['offset_mapping']):
                labels=align_token_labels(offsets,row['entities'],row.get('complete_annotation',False))
                if all(x==-100 for x in labels):continue
                examples.append({**{k:v[i] for k,v in enc.items() if k not in ('offset_mapping','overflow_to_sample_mapping')},'labels':labels})
        if not examples:raise ValueError('No usable annotated tokens')
        return Dataset.from_list(examples)
    train_ds,val_ds=dataset(train_rows),dataset(val_rows)
    model=AutoModelForTokenClassification.from_pretrained(a.base,num_labels=len(TAGS),
                id2label=dict(enumerate(TAGS)),label2id={t:i for i,t in enumerate(TAGS)})
    # No misleading entity F1 over partially annotated data; report token accuracy
    # on observed labels and held-out loss. Full-document evaluation is separate.
    def metrics(result):
        pred=np.argmax(result.predictions,axis=-1);mask=result.label_ids!=-100
        return {'observed_token_accuracy':float((pred[mask]==result.label_ids[mask]).mean())}
    args=TrainingArguments(output_dir=a.output,num_train_epochs=a.epochs,learning_rate=3e-5,
        per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=a.batch_size,
        warmup_ratio=.1,weight_decay=.01,eval_strategy='epoch',save_strategy='epoch',
        load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
        seed=a.seed,report_to='none',fp16=torch.cuda.is_available(),save_total_limit=2)
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),processing_class=tokenizer,compute_metrics=metrics)
    trainer.train();trainer.save_model(a.output);tokenizer.save_pretrained(a.output)
    write_json(Path(a.output)/'training_provenance.json',{'base':a.base,'train':a.train,'validation':a.validation,
         'seed':a.seed,'weak_training':any(not r.get('complete_annotation') for r in train_rows),
         'warning':'Positive-only weak alignment lacks reliable O supervision; review complete gold annotations for robust NER.',
         'validation_metrics':trainer.evaluate()})

if __name__=='__main__':main()
