"""Entity-span metrics on fully annotated documents only."""
def bio_spans(tags):
    spans=set(); start=None; current=None
    for i,tag in enumerate(list(tags)+['O']):
        kind=tag[2:] if tag.startswith(('B-','I-')) else None
        if start is not None and (not tag.startswith('I-') or kind!=current):
            spans.add((start,i,current));start=None
        if kind and start is None:
            start=i;current=kind
    return spans

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'tp':tp,'fp':fp,'fn':fn}

def span_metrics(predictions, references):
    tp=fp=fn=0
    for pred,truth in zip(predictions,references):
        p={(e['start'],e['end'],e['label']) for e in pred}
        t={(e['start'],e['end'],e['label']) for e in truth}
        tp+=len(p&t);fp+=len(p-t);fn+=len(t-p)
    return prf(tp,fp,fn)
