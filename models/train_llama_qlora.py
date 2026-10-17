"""Completion-only NF4 QLoRA training with actual cached NER and retrieved cases."""
import argparse
from pathlib import Path
from clinical.io import read_records,write_json,fingerprint
from clinical.prompts import completion_example,CompletionCollator

BASE_MODEL='meta-llama/Meta-Llama-3-8B-Instruct'

def get_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('NF4 QLoRA requires a supported NVIDIA CUDA GPU; no silent CPU fallback')
    return BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

def get_qlora_config():
    from peft import LoraConfig,TaskType
    return LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,bias='none',task_type=TaskType.CAUSAL_LM,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])

def main():
    p=argparse.ArgumentParser()
    for name in ('train','validation','index'):p.add_argument('--'+name,required=True)
    p.add_argument('--output',default='checkpoints/llama-qlora');p.add_argument('--base',default=BASE_MODEL)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--gradient-accumulation',type=int,default=16);p.add_argument('--max-length',type=int,default=8192)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--allow-in-sample-entities',action='store_true')
    a=p.parse_args()
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,set_seed
    from peft import get_peft_model,prepare_model_for_kbit_training
    from models.rag_retriever import ClinicalRAGRetriever
    set_seed(a.seed)
    train,val=read_records(a.train),read_records(a.validation)
    for row in train+val:
        if 'entities' not in row or not row.get('entity_source'):
            raise ValueError('Actual NER prediction cache required; run scripts.predict_entities or scripts.oof_entities')
    if not a.allow_in_sample_entities and any(not r.get('out_of_fold') for r in train):
        raise ValueError('Training needs out-of-fold NER predictions. For an explicitly documented approximation pass --allow-in-sample-entities.')
    if {fingerprint(r['note']) for r in train}&{fingerprint(r['note']) for r in val}:
        raise ValueError('Training/validation note leakage')
    retriever=ClinicalRAGRetriever.load(a.index)
    if not {str(r['id']) for r in retriever.rows}<={str(r['id']) for r in train}:
        raise ValueError('Retrieval index must contain training examples only')
    tokenizer=AutoTokenizer.from_pretrained(a.base,use_fast=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    def build(rows):
        items=[];used=[]
        for row in rows:
            cases=retriever.retrieve(row['note'],top_k=3,exclude_id=row['id'])
            item,k=completion_example(tokenizer,row['note'],row['entities'],cases,row['target'],a.max_length)
            items.append(item);used.append(k)
        return Dataset.from_list(items),used
    train_ds,train_k=build(train);val_ds,val_k=build(val)
    model=AutoModelForCausalLM.from_pretrained(a.base,quantization_config=get_bnb_config(),device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model);model=get_peft_model(model,get_qlora_config())
    model.config.use_cache=False;model.print_trainable_parameters()
    args=TrainingArguments(output_dir=a.output,num_train_epochs=a.epochs,learning_rate=2e-4,
        per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.gradient_accumulation,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},optim='paged_adamw_8bit',
        bf16=torch.cuda.is_bf16_supported(),fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='eval_loss',greater_is_better=False,save_total_limit=2,
        warmup_ratio=.03,lr_scheduler_type='cosine',report_to='none',seed=a.seed)
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,
        data_collator=CompletionCollator(tokenizer.pad_token_id),processing_class=tokenizer)
    trainer.train();trainer.save_model(a.output);tokenizer.save_pretrained(a.output)
    write_json(Path(a.output)/'training_provenance.json',{'base':a.base,'seed':a.seed,
        'in_sample_ner_approximation':a.allow_in_sample_entities,'train':a.train,'validation':a.validation,
        'index':a.index,'max_length':a.max_length,'examples_kept_train':train_k,'examples_kept_validation':val_k,
        'validation_metrics':trainer.evaluate()})

if __name__=='__main__':main()
