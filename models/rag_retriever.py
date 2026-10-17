"""Explicit embedding model + FAISS cosine retrieval with training provenance."""
import argparse
import json
from pathlib import Path
from clinical.io import read_records,write_json, fingerprint

class ClinicalRAGRetriever:
    def __init__(self,model_name='BAAI/bge-large-en-v1.5',device='cpu'):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.faiss,self.np=faiss,np
        self.model_name=model_name
        self.model=SentenceTransformer(model_name,device=device)
        self.index=faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        self.rows=[]
    def _embed(self,texts,query=False):
        # BGE retrieval instruction applies only to queries, never corpus notes.
        if query and self.model_name.startswith('BAAI/bge-'):
            texts=['Represent this sentence for searching relevant passages: '+s for s in texts]
        return self.np.asarray(self.model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=False),dtype='float32')
    def build_index(self,records):
        if not records:raise ValueError('Empty retrieval corpus')
        if any('target' not in r for r in records):raise ValueError('Retrieval examples need labels')
        self.rows=records;self.index.reset();self.index.add(self._embed([r['note'] for r in records]))
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.faiss.write_index(self.index,str(path/'index.faiss'))
        write_json(path/'metadata.json',{'model_name':self.model_name,'rows':self.rows,
            'long_notes':'Embedding encoder truncates at its max_seq_length; generation still receives the full query note.',
            'embedding_max_tokens':self.model.max_seq_length})
    @classmethod
    def load(cls,path,device='cpu'):
        path=Path(path);meta=json.loads((path/'metadata.json').read_text())
        obj=cls(meta['model_name'],device);obj.rows=meta['rows'];obj.index=obj.faiss.read_index(str(path/'index.faiss'))
        if obj.index.d!=obj.model.get_sentence_embedding_dimension() or obj.index.ntotal!=len(obj.rows):
            raise ValueError('Corrupt/incompatible FAISS metadata')
        return obj
    def retrieve(self,note,top_k=3,exclude_id=None):
        if not self.rows:return []
        if top_k<1:raise ValueError('top_k must be positive')
        # Overfetch all so duplicate exclusion cannot silently reduce k.
        scores,indices=self.index.search(self._embed([note],query=True),len(self.rows))
        from clinical.data import near_signature
        query_sig=near_signature(note);seen={fingerprint(note)};results=[]
        for score,idx in zip(scores[0],indices[0]):
            if idx<0:continue
            row=self.rows[int(idx)];fp=fingerprint(row['note'])
            if (exclude_id is not None and str(row['id'])==str(exclude_id)) or fp in seen or near_signature(row['note'])==query_sig:continue
            seen.add(fp);results.append({**row,'score':float(score)})
            if len(results)==top_k:break
        return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--output',default='artifacts/retriever');p.add_argument('--model',default='BAAI/bge-large-en-v1.5')
    a=p.parse_args();r=ClinicalRAGRetriever(a.model);r.build_index(read_records(a.train));r.save(a.output)
