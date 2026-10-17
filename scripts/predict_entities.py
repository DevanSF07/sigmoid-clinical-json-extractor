"""Cache checkpoint predictions for LLM prompts; never replace with JSON targets."""
import argparse
from clinical.io import read_records,write_jsonl
from models.ner_bioclinicalbert import BioClinicalBERTNER

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();model=BioClinicalBERTNER(a.checkpoint)
    rows=[]
    for row in read_records(a.input):
        rows.append({**row,'entities':model.extract_entities(row['note']),'entity_source':a.checkpoint})
    write_jsonl(a.output,rows)
if __name__=='__main__':main()
