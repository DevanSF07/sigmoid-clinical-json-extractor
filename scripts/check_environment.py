"""Report runtime readiness without downloading weights or printing credentials."""
import importlib.metadata
import importlib.util
import json

def main():
    versions={}
    for pkg in ['torch','transformers','peft','datasets','sentence-transformers','faiss-cpu','outlines','pydantic','sentencepiece']:
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:versions[pkg]=None
    report={'versions':versions,'cuda_available':False,'device':None}
    if importlib.util.find_spec('torch'):
        import torch
        report['cuda_available']=torch.cuda.is_available()
        if report['cuda_available']:report['device']=torch.cuda.get_device_name()
    report['nf4_training_ready']=report['cuda_available'] and importlib.util.find_spec('bitsandbytes') is not None
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
