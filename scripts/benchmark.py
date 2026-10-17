"""One explicit pipeline configuration per run; compare saved reports fairly."""
import argparse
import platform
import statistics
import time
from clinical.io import read_records,write_json
from pipeline import ClinicalExtractionPipeline

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ner-checkpoint',required=True);p.add_argument('--index',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--no-quantization',action='store_true');p.add_argument('--unconstrained',action='store_true')
    p.add_argument('--samples',type=int,default=30);p.add_argument('--warmup',type=int,default=2)
    a=p.parse_args()
    import torch
    rows=read_records(a.input)[:a.samples]
    if not rows:raise ValueError('Empty benchmark input')
    model=ClinicalExtractionPipeline('model',a.ner_checkpoint,a.index,a.adapter,not a.no_quantization,not a.unconstrained)
    def sync():
        if torch.cuda.is_available():torch.cuda.synchronize()
    for i in range(a.warmup):model.process(rows[i%len(rows)]['note'])
    sync()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    durations=[];failures=0;prompt_tokens=[];output_tokens=[]
    for row in rows:
        sync();t=time.perf_counter();result=model.process(row['note'],row['id']);sync()
        durations.append(time.perf_counter()-t);failures+=result['record'] is None
        prompt_tokens.append(result['metadata'].get('prompt_tokens'))
        output_tokens.append(len(model.generator.tokenizer(result['raw'],add_special_tokens=False)['input_ids']) if result['raw'] else 0)
    ordered=sorted(durations)
    write_json(a.output,{'hardware':torch.cuda.get_device_name() if torch.cuda.is_available() else platform.processor(),
        'torch':torch.__version__,'configuration':vars(a),'batch_size':1,'latency_mean_seconds':statistics.mean(durations),
        'latency_p95_seconds':ordered[min(len(ordered)-1,int(.95*len(ordered)))],
        'notes_per_second':len(rows)/sum(durations),'failures':failures,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        'input_prompt_token_counts':prompt_tokens,'output_token_counts':output_tokens,
        'note':'Compare identical notes, adapter, context, generation and hardware; model initialization excluded. No speedup assumed.'})
if __name__=='__main__':main()
