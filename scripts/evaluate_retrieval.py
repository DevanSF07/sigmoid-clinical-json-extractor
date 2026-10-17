import argparse
from clinical.io import read_records,write_json
from models.rag_retriever import ClinicalRAGRetriever

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--index',required=True);p.add_argument('--output',default='artifacts/retrieval_metrics.json')
    a=p.parse_args();rows=read_records(a.input);r=ClinicalRAGRetriever.load(a.index)
    if not rows:raise ValueError('Empty evaluation set')
    hits=rr=0
    for row in rows:
        cases=r.retrieve(row['note'],3,exclude_id=row['id'])
        ranks=[i for i,c in enumerate(cases,1) if c['target']['visit_motivation']==row['target']['visit_motivation']]
        if ranks:hits+=1;rr+=1/ranks[0]
    write_json(a.output,{'Hit@3':hits/len(rows),'MRR@3':rr/len(rows),'queries':len(rows),
        'relevance':'Same visit motivation: category proxy, not clinician relevance judgments',
        'embedding_model':r.model_name})
if __name__=='__main__':main()
