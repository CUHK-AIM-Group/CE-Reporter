import torch
import json
import os
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(model_name, lora_path, gpu_id):
    """Setup model and tokenizer with LoRA adapter"""
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.empty_cache()

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, device_map={"": 0}, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, trust_remote_code=True, rope_scaling=None)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path, device_map={"": 0})
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=2048):
    """Generate response using the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id) 
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def answer_template(prompt):
    """Create answer template"""
    return f"User: {prompt}\n\nAssistant:"

def main():
    """Main inference function with command line arguments"""
    parser = argparse.ArgumentParser(description='Generate responses using fine-tuned LLM')
    parser.add_argument('--model_name', type=str,
                       default="DeepSeek-R1-Distill-Llama-8B",
                       help='Path to base model')
    parser.add_argument('--lora_path', type=str,
                       required=True,
                       help='Path to LoRA adapter')
    parser.add_argument('--gpu_id', type=int,
                       default=0,
                       help='GPU ID to use')
    parser.add_argument('--input_file', type=str,
                       default="llava_caption_sum.json",
                       help='Path to input JSON file')
    parser.add_argument('--output_file', type=str,
                       default="ce-report-text.json",
                       help='Path to output JSON file')
    parser.add_argument('--max_length', type=int,
                       default=2048,
                       help='Maximum generation length')
    
    args = parser.parse_args()

    # Setup model and tokenizer
    model, tokenizer = setup_model(args.model_name, args.lora_path, args.gpu_id)

    # Load test data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Process all samples
    processed_samples = []

    for sample in test_data:
        instruction = sample['instruction']
        input_text = sample['input']
        input_all = f"{instruction} {input_text}"
        sample['input'] = "None"
        
        # Generate response
        response = generate_response(model, tokenizer, answer_template(input_all), args.max_length)
        output_value = response.split("Assistant: ")[-1]
        sample['output'] = output_value
        
        print("Finished one sample")
        processed_samples.append(sample)

    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        json.dump(processed_samples, f_out, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()