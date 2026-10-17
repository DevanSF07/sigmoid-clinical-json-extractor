"""CPU smoke: fit/save/reload tiny random models, no downloaded clinical weights."""
import pytest

pytestmark=pytest.mark.integration

def test_tiny_ner_training_and_reload(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('transformers');pytest.importorskip('datasets')
    import subprocess
    import sys
    from transformers import BertConfig,BertForTokenClassification,BertTokenizerFast
    from clinical.annotations import TAGS
    from clinical.io import write_jsonl
    base=tmp_path/'base';base.mkdir()
    (base/'vocab.txt').write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','patient','has','cough','.']))
    tok=BertTokenizerFast(vocab_file=str(base/'vocab.txt'));tok.save_pretrained(base)
    model=BertForTokenClassification(BertConfig(vocab_size=9,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
        intermediate_size=32,num_labels=len(TAGS),id2label=dict(enumerate(TAGS)),label2id={t:i for i,t in enumerate(TAGS)}))
    model.save_pretrained(base)
    row={'id':'a','note':'Patient has cough.','entities':[{'start':12,'end':17,'label':'SYMPTOM'}],
         'complete_annotation':True,'annotation_source':'manual_gold'}
    write_jsonl(tmp_path/'train.jsonl',[row]);write_jsonl(tmp_path/'val.jsonl',[{**row,'id':'b'}])
    result=subprocess.run([sys.executable,'-m','models.train_ner','--train',str(tmp_path/'train.jsonl'),
        '--validation',str(tmp_path/'val.jsonl'),'--base',str(base),'--output',str(tmp_path/'out'),'--epochs','1','--batch-size','1'],
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    from models.ner_bioclinicalbert import BioClinicalBERTNER
    ner=BioClinicalBERTNER(tmp_path/'out')
    assert isinstance(ner.extract_entities('Patient has cough.'),list)
    assert (tmp_path/'out'/'training_provenance.json').exists()

def test_tiny_llama_lora_completion_training(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('peft');pytest.importorskip('datasets')
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast,LlamaConfig,LlamaForCausalLM,Trainer,TrainingArguments,AutoModelForCausalLM
    from datasets import Dataset
    from peft import get_peft_model,PeftModel
    from models.train_llama_qlora import get_qlora_config
    from clinical.prompts import completion_example,CompletionCollator
    raw=Tokenizer(models.WordLevel({'[PAD]':0,'[UNK]':1,'[EOS]':2,'system':3,'user':4,'assistant':5},unk_token='[UNK]'))
    raw.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]')
    tok.chat_template="{% for message in messages %}{{ message['role'] + '\n' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant\n' }}{% endif %}"
    row,_=completion_example(tok,'Patient reports cough.',[],[],{'patient_info':{},'symptoms':['cough'],'vital_signs':{}},max_length=4096)
    base=tmp_path/'base';adapter=tmp_path/'adapter'
    model=LlamaForCausalLM(LlamaConfig(vocab_size=6,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=4096,pad_token_id=0,bos_token_id=2,eos_token_id=2))
    model.save_pretrained(base)
    model=get_peft_model(AutoModelForCausalLM.from_pretrained(base),get_qlora_config())
    trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(tmp_path/'run'),max_steps=1,per_device_train_batch_size=1,
        report_to='none',save_strategy='no',use_cpu=True),train_dataset=Dataset.from_list([row]),data_collator=CompletionCollator(0))
    output=trainer.train()
    assert output.training_loss>0
    model.save_pretrained(adapter)
    restored=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base),adapter)
    assert restored.peft_config['default'].r==32
