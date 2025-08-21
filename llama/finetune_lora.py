import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
import os
import argparse

def setup_model_and_tokenizer(model_name, gpu_id):
    """Setup model and tokenizer with specified GPU"""
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.empty_cache()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True,
        quantization_config=quantization_config
    )

    return model, tokenizer

def create_lora_config():
    """Create LoRA configuration"""
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        init_lora_weights=True,
        inference_mode=False
    )
    return lora_config

def create_training_args(output_dir, num_train_epochs):
    """Create training arguments"""
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        optim="paged_adamw_32bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=3e-4,
        fp16=True,
        bf16=False,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
    )
    return training_arguments

def process_func(example):
    """Process data for training"""
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"User: {example['instruction']} {example['input']}\n\n", add_special_tokens=False)
    response = tokenizer(f"Assistant: {example['output']}{tokenizer.eos_token}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def load_and_process_data(data_path):
    """Load and process training data"""
    df = pd.read_json(data_path)
    ds = Dataset.from_pandas(df)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    return tokenized_id

def main():
    """Main training function with command line arguments"""
    parser = argparse.ArgumentParser(description='Train LLM with LoRA fine-tuning')
    parser.add_argument('--model_name', type=str, 
                       default="DeepSeek-R1-Distill-Llama-8B",
                       help='Path to model directory')
    parser.add_argument('--gpu_id', type=int, 
                       default=0,
                       help='GPU ID to use')
    parser.add_argument('--data_path', type=str,
                       default='../vce_data/llama/llama_annotations.json',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str,
                       default="ce-reporter-llama3_1_8b_LoRA",
                       help='Output directory for trained model')
    parser.add_argument('--num_epochs', type=int,
                       default=8,
                       help='Number of training epochs')
    
    args = parser.parse_args()

    # Setup model and tokenizer
    global tokenizer  # Make tokenizer available to process_func
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.gpu_id)

    # Apply LoRA configuration
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # Create training arguments
    training_args = create_training_args(args.output_dir, args.num_epochs)

    # Load and process data
    tokenized_data = load_and_process_data(args.data_path)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()