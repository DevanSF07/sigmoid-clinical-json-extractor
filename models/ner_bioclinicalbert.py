"""Real sliding-window BioClinicalBERT token-classification inference."""
from pathlib import Path
from clinical.annotations import TAGS
from clinical.assertions import assertion_status

class BioClinicalBERTNER:
    def __init__(self, checkpoint, device=None):
        if not Path(checkpoint,'config.json').exists():
            raise ValueError('NER requires a fine-tuned local checkpoint, not untrained base weights')
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.torch=torch
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,use_fast=True)
        self.model=AutoModelForTokenClassification.from_pretrained(checkpoint).to(self.device).eval()
        if [self.model.config.id2label[i] for i in range(len(TAGS))]!=TAGS:
            raise ValueError('NER checkpoint label mapping does not match project labels')

    def extract_entities(self,note):
        encoded=self.tokenizer(note,return_offsets_mapping=True,return_overflowing_tokens=True,
                               truncation=True,max_length=512,stride=96,padding=True,return_tensors='pt')
        offsets=encoded.pop('offset_mapping').tolist();encoded.pop('overflow_to_sample_mapping',None)
        # Merge window predictions by selecting the most confident prediction per original token.
        best={}
        with self.torch.inference_mode():
            for i in range(0,len(offsets),8):
                batch={k:v[i:i+8].to(self.device) for k,v in encoded.items()}
                probs=self.model(**batch).logits.softmax(-1).cpu()
                scores,ids=probs.max(-1)
                for wi in range(len(scores)):
                    for ti,(s,e) in enumerate(offsets[i+wi]):
                        if s==e:continue
                        key=(s,e); conf=float(scores[wi,ti]);tag=TAGS[int(ids[wi,ti])]
                        if key not in best or conf>best[key][1]:best[key]=(tag,conf)
        entities=[];active=None
        for (s,e),(tag,score) in sorted(best.items()):
            label=tag[2:] if tag!='O' else None
            continuation=active is not None and tag.startswith('I-') and active['label']==label and not note[active['end']:s].strip()
            if not continuation and active:
                entities.append(active);active=None
            if label:
                if continuation:
                    active['end']=e;active['_scores'].append(score)
                else:active={'start':s,'end':e,'label':label,'_scores':[score]}
        if active:entities.append(active)
        for ent in entities:
            ent['text']=note[ent['start']:ent['end']]
            ent['confidence']=sum(ent['_scores'])/len(ent.pop('_scores'))
            ent['assertion']=assertion_status(note,ent['start'],ent['end'])
        return entities
