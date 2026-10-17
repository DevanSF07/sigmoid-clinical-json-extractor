"""Load a real adapter and use Outlines v1 token constraints. No fallback path."""
from pathlib import Path
from clinical.prompts import fit_prompt
from schemas.patient_schema import PatientRecord

class OutlinesSchemaGenerator:
    def __init__(self,adapter,quantized=True,context_tokens=8192,max_new_tokens=1024):
        import torch
        import outlines
        from outlines.types import JsonSchema
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import PeftConfig,PeftModel
        if not Path(adapter,'adapter_config.json').exists():raise ValueError('Fine-tuned LoRA adapter directory required')
        config=PeftConfig.from_pretrained(adapter)
        self.tokenizer=AutoTokenizer.from_pretrained(adapter)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        kwargs={'torch_dtype':torch.float32}
        if quantized:
            from models.train_llama_qlora import get_bnb_config
            kwargs.update(quantization_config=get_bnb_config(),device_map='auto',
                          torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        base=AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,**kwargs)
        self.model=PeftModel.from_pretrained(base,adapter).eval()
        self.model.config.use_cache=True
        self.generator=outlines.Generator(outlines.from_transformers(self.model,self.tokenizer),JsonSchema(PatientRecord,whitespace_pattern=r" ?"))
        self.unconstrained=outlines.Generator(outlines.from_transformers(self.model,self.tokenizer))
        self.max_new_tokens=max_new_tokens
        self.context_tokens=min(context_tokens,getattr(base.config,'max_position_embeddings',context_tokens))
        if self.context_tokens<=max_new_tokens:raise ValueError('Generation reserve exceeds context')
    def generate(self,note,entities,cases,constrained=True):
        prompt,msgs,kept=fit_prompt(self.tokenizer,note,entities,cases,self.context_tokens-self.max_new_tokens)
        # Return raw text so evaluation can count invalid generations before repair.
        generator=self.generator if constrained else self.unconstrained
        raw=generator(prompt,max_new_tokens=self.max_new_tokens,do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id)
        return raw,{'retrieved_examples_used':kept,'constrained':constrained,
            'prompt_tokens':len(self.tokenizer(prompt,add_special_tokens=False)['input_ids'])}
