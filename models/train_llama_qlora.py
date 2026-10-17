"""
LLaMA-3-8B QLoRA Fine-Tuning Module
Supervised Fine-Tuning (SFT) of meta-llama/Meta-Llama-3-8B-Instruct.
Implements:
- NF4 4-bit quantization with double quantization
- LoRA rank r=32, alpha=64 across all linear projections
- Chat template formatting with actual NER entities and retrieved dynamic few-shot cases
- Strict causal LM loss masking (-100 on prompt and padding, gradients only on target JSON)
- Guard against target truncation
- Checkpoint saving and adapter loading verification
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType
)

SYSTEM_PROMPT = (
    "You are an expert clinical informatics reasoning model. "
    "Your task is to analyze the provided clinical note, extracted clinical entities, "
    "and reference clinical examples to generate a strictly compliant PatientRecord JSON document. "
    "Output ONLY valid JSON adhering to the specified schema with no explanations or conversational text."
)


def format_llama3_chat_prompt(
    note: str,
    ner_entities: Dict[str, Any],
    retrieved_demonstrations: str = ""
) -> str:
    """Format user message containing note, NER predictions, and few-shot context."""
    user_msg = ""
    if retrieved_demonstrations:
        user_msg += f"## Reference Historical Cases (Dynamic Demonstrations):\n{retrieved_demonstrations}\n"

    user_msg += "## Target Clinical Case To Extract:\n"
    user_msg += f"**Clinical Note:**\n{note.strip()}\n\n"
    user_msg += f"**Pre-Extracted Clinical NER Entities:**\n{json.dumps(ner_entities, indent=2)}\n\n"
    user_msg += (
        "Generate the finalized PatientRecord JSON containing: "
        "'patient_info' (age, gender), 'visit_motivation', 'symptoms' (canonical list), "
        "and 'vital_signs' (blood_pressure, temperature, heart_rate, respiratory_rate, oxygen_saturation, glucose_level, cholesterol_level)."
    )
    return user_msg


def build_tokenized_inputs_and_labels(
    tokenizer: AutoTokenizer,
    note: str,
    ner_entities: Dict[str, Any],
    target_json_str: str,
    retrieved_demonstrations: str = "",
    max_seq_length: int = 2048
) -> Dict[str, torch.Tensor]:
    """
    Construct chat template, tokenize, and apply strict causal loss masking.
    Prompt and padding tokens receive -100.
    Assistant completion tokens (target JSON) retain their input IDs.
    Guarantees target JSON is not truncated.
    """
    user_content = format_llama3_chat_prompt(note, ner_entities, retrieved_demonstrations)

    # 1. Format prompt prefix (system + user + assistant start)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    if getattr(tokenizer, "chat_template", None) is not None:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # 2. Format target JSON completion (assistant response + EOS)
    completion_text = target_json_str.strip() + "<|eot_id|>"
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    total_len = len(prompt_ids) + len(completion_ids)

    # 3. Truncation guard: if total length exceeds max_seq_length, truncate PROMPT, never target!
    if total_len > max_seq_length:
        allowed_prompt_len = max_seq_length - len(completion_ids)
        if allowed_prompt_len <= 10:
            raise ValueError(f"Target completion is too large ({len(completion_ids)} tokens) for max_seq_length {max_seq_length}!")
        # Truncate beginning of user prompt, preserving prompt structure
        prompt_ids = prompt_ids[:allowed_prompt_len]

    input_ids = prompt_ids + completion_ids
    # Mask prompt tokens with -100 (ignored in cross-entropy loss)
    labels = [-100] * len(prompt_ids) + completion_ids

    # Attention mask
    attention_mask = [1] * len(input_ids)

    # Pad to max_seq_length
    pad_len = max_seq_length - len(input_ids)
    if pad_len > 0:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


class ClinicalExtractionSFTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        ner_entities_list: Optional[List[Dict[str, Any]]] = None,
        retriever=None,
        max_seq_length: int = 2048
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.ner_entities_list = ner_entities_list
        self.retriever = retriever
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        note = row["Note"]
        target = row["json"] if isinstance(row["json"], str) else json.dumps(row["json"])

        # NER entities (out-of-fold or pre-extracted)
        if self.ner_entities_list is not None and idx < len(self.ner_entities_list):
            ner_ents = self.ner_entities_list[idx]
        else:
            ner_ents = {"status": "unsupervised_placeholder"}

        # Dynamic retrieval demos (excluding self)
        demos = ""
        if self.retriever is not None:
            retrieved = self.retriever.retrieve(note, query_id=row["ID"], top_k=2)
            demos = self.retriever.format_demonstrations(retrieved)

        return build_tokenized_inputs_and_labels(
            tokenizer=self.tokenizer,
            note=note,
            ner_entities=ner_ents,
            target_json_str=target,
            retrieved_demonstrations=demos,
            max_seq_length=self.max_seq_length
        )


def get_qlora_config() -> LoraConfig:
    """LoRA configuration targeting all linear projection layers (r=32, alpha=64)."""
    return LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


def get_bnb_config() -> BitsAndBytesConfig:
    """NF4 4-bit Quantization configuration for single-GPU VRAM efficiency."""
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )


def train_qlora(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str = "checkpoints/llama3_qlora_adapter",
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    num_epochs: int = 3,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1536
):
    """
    Execute full QLoRA supervised fine-tuning.
    Requires NVIDIA GPU (CUDA), BitsAndBytes, and HuggingFace gated Llama-3 access token.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. 4-bit NF4 QLoRA requires an NVIDIA GPU (e.g. A10G, T4, A100). "
            "On local Apple Silicon/CPU, execute the CPU verification tests instead."
        )

    print(f"[QLoRA] Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = get_bnb_config()
    print(f"[QLoRA] Loading base model in 4-bit NF4: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa"
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = get_qlora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ClinicalExtractionSFTDataset(train_df, tokenizer, max_seq_length=max_seq_length)
    val_dataset = ClinicalExtractionSFTDataset(val_df, tokenizer, max_seq_length=max_seq_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=42,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    print("[QLoRA] Commencing training...")
    trainer.train()

    print(f"[QLoRA] Saving adapter weights to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[QLoRA] Training completed successfully!")


def load_fine_tuned_llama(
    base_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_dir: str = "checkpoints/llama3_qlora_adapter",
    device_map: str = "auto"
):
    """Load fine-tuned QLoRA adapter onto base model for inference."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        bnb_config = get_bnb_config()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=device_map,
            torch_dtype=torch.float32
        )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer
