"""Evaluate saved raw outputs without needing to reload any model."""
import argparse
import json
from clinical.io import read_records,write_json
from clinical.evaluation import extraction_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--truth',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='artifacts/extraction_metrics.json');p.add_argument('--atol',type=float,default=.01)
    a=p.parse_args()
    with open(a.predictions) as f:predictions=[json.loads(s) for s in f if s.strip()]
    result=extraction_metrics(read_records(a.truth),predictions,a.atol)
    write_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
