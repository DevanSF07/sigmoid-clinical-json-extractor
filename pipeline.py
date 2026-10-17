"""Explicit rules baseline or fully integrated NER→RAG→QLoRA→Outlines pipeline."""
import argparse
import json
import re
import time
from pathlib import Path
from clinical.io import read_records,write_jsonl
from clinical.taxonomy import MOTIVATION_MAPPING
from clinical.assertions import assertion_status
from schemas.patient_schema import PatientRecord


def extract_motivation(note):
    clean=note.replace('*','')
    headers=list(re.finditer(r'(?:chief complaint|visit motivation|reason for visit|assessment|diagnosis)\s*:\s*([^\n]+)',clean,re.I))
    for header in headers:
        for category,words in MOTIVATION_MAPPING.items():
            for word in sorted(words,key=len,reverse=True):
                m=re.search(r'\b'+re.escape(word)+r'\b',header.group(1),re.I)
                if m and assertion_status(clean,header.start(1)+m.start(),header.start(1)+m.end())=='present':return category
    return None

class ClinicalExtractionPipeline:
    def __init__(self,mode='model',ner_checkpoint=None,index=None,adapter=None,quantized=True,
                 constrained=True,context_tokens=8192,max_new_tokens=1024,
                 ner=None,retriever=None,generator=None):
        if mode not in ('rules','model'):raise ValueError('mode must be rules or model')
        self.mode=mode;self.constrained=constrained
        if mode=='rules':
            from models.rules import RuleExtractor
            self.rules=RuleExtractor()
        else:
            if ner is None:
                from models.ner_bioclinicalbert import BioClinicalBERTNER
                ner=BioClinicalBERTNER(ner_checkpoint)
            if retriever is None:
                from models.rag_retriever import ClinicalRAGRetriever
                retriever=ClinicalRAGRetriever.load(index)
            if generator is None:
                from models.outlines_generator import OutlinesSchemaGenerator
                generator=OutlinesSchemaGenerator(adapter,quantized,context_tokens,max_new_tokens)
            self.ner,self.retriever,self.generator=ner,retriever,generator
    def process(self,note,record_id=None):
        start=time.perf_counter();meta={'mode':self.mode}
        try:
            if self.mode=='rules':
                raw=self.rules.extract_entities(note)
                raw['visit_motivation']=extract_motivation(note)
                raw=json.dumps(raw)
            else:
                t=time.perf_counter();entities=self.ner.extract_entities(note);meta['ner_seconds']=time.perf_counter()-t
                t=time.perf_counter();cases=self.retriever.retrieve(note,top_k=3,exclude_id=record_id)
                meta['retrieval_seconds']=time.perf_counter()-t
                t=time.perf_counter();raw,details=self.generator.generate(note,entities,cases,constrained=self.constrained)
                meta.update(details);meta['generation_seconds']=time.perf_counter()-t
                meta['retrieved_ids']=[r['id'] for r in cases]
            result={'id':record_id,'raw':raw,'record':None,'error':None}
            try:parsed=json.loads(raw);meta['json_valid']=True
            except (ValueError,TypeError) as e:
                meta.update(json_valid=False,schema_valid=False);result['error']=str(e)
            else:
                try:
                    record=PatientRecord.model_validate(parsed)
                    result['record']=record.to_clean_dict();meta['schema_valid']=True
                except Exception as e:
                    meta['schema_valid']=False;result['error']=str(e)
        except Exception as e:
            result={'id':record_id,'raw':None,'record':None,'error':f'{type(e).__name__}: {e}'}
            meta.update(json_valid=False,schema_valid=False)
        meta['seconds']=time.perf_counter()-start
        result['metadata']=meta
        return result
    def process_note(self,note):
        result=self.process(note)
        if result['error']:raise ValueError(result['error'])
        return result['record']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['rules','model'],default='model')
    inp=p.add_mutually_exclusive_group(required=True);inp.add_argument('--input');inp.add_argument('--note')
    p.add_argument('--output',default='artifacts/predictions.jsonl');p.add_argument('--submission')
    p.add_argument('--ner-checkpoint');p.add_argument('--index');p.add_argument('--adapter')
    p.add_argument('--no-quantization',action='store_true');p.add_argument('--unconstrained',action='store_true')
    p.add_argument('--context-tokens',type=int,default=8192);p.add_argument('--max-new-tokens',type=int,default=1024)
    a=p.parse_args()
    if a.mode=='model' and not all([a.ner_checkpoint,a.index,a.adapter]):p.error('model mode requires --ner-checkpoint, --index, --adapter')
    pipeline=ClinicalExtractionPipeline(a.mode,a.ner_checkpoint,a.index,a.adapter,not a.no_quantization,
        not a.unconstrained,a.context_tokens,a.max_new_tokens)
    rows=read_records(a.input) if a.input else [{'id':'single','note':a.note}]
    # Stream progress and outputs to avoid losing an entire long inference run.
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    failures=0;results=[]
    with path.open('w') as f:
        for i,row in enumerate(rows,1):
            result=pipeline.process(row['note'],row['id']);results.append(result)
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
            failures+=result['record'] is None
            if i%100==0:print(f'Processed {i}/{len(rows)}; failures={failures}',flush=True)
    if a.submission:
        if failures:raise RuntimeError('Submission not written: failed records exist; inspect predictions JSONL')
        import csv
        target=Path(a.submission);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['ID','json']);writer.writeheader()
            writer.writerows({'ID':r['id'],'json':json.dumps(r['record'],ensure_ascii=False)} for r in results)
    print(json.dumps({'records':len(rows),'failures':failures,'output':str(path)}))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
