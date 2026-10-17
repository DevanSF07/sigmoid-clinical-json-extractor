"""Offline tiny random-weight interface tests, not trained clinical validation."""
import importlib.util
import json
import pytest
from pathlib import Path

pytestmark=pytest.mark.integration

@pytest.fixture
def tiny_bert(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('transformers')
    import torch
    from transformers import BertTokenizerFast,BertForTokenClassification,BertConfig
    from clinical.annotations import TAGS
    vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','patient','has','cough','.']
    (tmp_path/'vocab.txt').write_text('\n'.join(vocab))
    tokenizer=BertTokenizerFast(vocab_file=str(tmp_path/'vocab.txt'),do_lower_case=True)
    tokenizer.save_pretrained(tmp_path)
    config=BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
        intermediate_size=32,num_labels=len(TAGS),id2label=dict(enumerate(TAGS)),label2id={t:i for i,t in enumerate(TAGS)})
    model=BertForTokenClassification(config)
    with torch.no_grad():model.classifier.weight.zero_();model.classifier.bias.zero_();model.classifier.bias[0]=10
    model.save_pretrained(tmp_path)
    return tmp_path

def test_real_bert_load_and_overflow(tiny_bert):
    from models.ner_bioclinicalbert import BioClinicalBERTNER
    model=BioClinicalBERTNER(tiny_bert)
    assert model.extract_entities('Patient has cough. '*180)==[]

def test_collator_masks_padding_not_real_eos():
    pytest.importorskip('torch')
    from clinical.prompts import CompletionCollator
    result=CompletionCollator(2)([
        {'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,2]},
        {'input_ids':[1,5,2],'attention_mask':[1,1,1],'labels':[-100,5,2]}])
    assert result['labels'].tolist()==[[-100,2,-100],[-100,5,2]]
    assert result['attention_mask'].tolist()==[[1,1,0],[1,1,1]]

def test_actual_outlines_transformers_interface(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('outlines');pytest.importorskip('transformers')
    import torch
    import outlines
    from pydantic import BaseModel
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders
    from transformers import PreTrainedTokenizerFast,GPT2Config,GPT2LMHeadModel
    # A character vocabulary can express every required JSON character locally.
    chars=['[PAD]','[UNK]','[EOS]']+list(dict.fromkeys('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]":, .\n-'))
    raw=Tokenizer(models.WordLevel({c:i for i,c in enumerate(chars)},unk_token='[UNK]'))
    raw.pre_tokenizer=pre_tokenizers.Split('',behavior='isolated');raw.decoder=decoders.Fuse()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]')
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(chars),n_embd=16,n_layer=1,n_head=2,n_positions=128,
        bos_token_id=2,eos_token_id=2,pad_token_id=0)).eval()
    with torch.no_grad():model.lm_head.weight.zero_()
    class Flag(BaseModel):
        ok: bool
    generator=outlines.Generator(outlines.from_transformers(model,tok),Flag)
    result=generator('a',max_new_tokens=60,do_sample=False,pad_token_id=0)
    assert isinstance(Flag.model_validate_json(result).ok,bool)

def test_real_sentence_transformer_faiss_save_reload(tiny_bert,tmp_path):
    pytest.importorskip('sentence_transformers');pytest.importorskip('faiss')
    from sentence_transformers import SentenceTransformer,models
    from models.rag_retriever import ClinicalRAGRetriever
    embedder=SentenceTransformer(modules=[models.Transformer(str(tiny_bert)),models.Pooling(16)])
    embed_path=tmp_path/'embedder';embedder.save(str(embed_path))
    retriever=ClinicalRAGRetriever(str(embed_path))
    retriever.build_index([{'id':'a','note':'Patient has cough.','target':{}},
                           {'id':'b','note':'cough cough.','target':{}}])
    retriever.save(tmp_path/'index')
    loaded=ClinicalRAGRetriever.load(tmp_path/'index')
    results=loaded.retrieve('Patient has cough.',3,exclude_id='a')
    assert [r['id'] for r in results]==['b']
    assert loaded.index.ntotal==2

def test_actual_adapter_load_and_patient_schema_compile(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('peft');pytest.importorskip('outlines')
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders
    from transformers import PreTrainedTokenizerFast,GPT2Config,GPT2LMHeadModel,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    from models.outlines_generator import OutlinesSchemaGenerator
    chars=['[PAD]','[UNK]','[EOS]']+list(dict.fromkeys('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]":, .\n-%°/_()'))
    raw=Tokenizer(models.WordLevel({c:i for i,c in enumerate(chars)},unk_token='[UNK]'))
    raw.pre_tokenizer=pre_tokenizers.Split('',behavior='isolated');raw.decoder=decoders.Fuse()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]')
    base=tmp_path/'base';adapter=tmp_path/'adapter'
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(chars),n_embd=16,n_layer=1,n_head=2,n_positions=128,
        bos_token_id=2,eos_token_id=2,pad_token_id=0))
    model.save_pretrained(base)
    model=AutoModelForCausalLM.from_pretrained(base)
    adapted=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'))
    adapted.save_pretrained(adapter);tok.save_pretrained(adapter)
    generator=OutlinesSchemaGenerator(str(adapter),quantized=False,context_tokens=128,max_new_tokens=64)
    assert generator.generator is not None
    assert generator.model.peft_config
