import json
import os
import argparse

def read_jsonl(file_path):
    """Read data from jsonl file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def generate_new_jsonl(test_img_file, answer_file, output_file):
    """Generate new jsonl file with structured medical report data"""
    test_img_data = read_jsonl(test_img_file)
    answer_data = read_jsonl(answer_file)

    # Create dictionary to store case information
    case_dict = {}

    for img_entry, answer_entry in zip(test_img_data, answer_data):
        question_id = img_entry['question_id']
        case_name = img_entry['image'].split('/')[-2]  
        time_str = img_entry['image'].split('_')[-1].replace('.jpg', '').replace("-", ":")
        text = answer_entry['text']

        # Generate image information
        img_info = f"'{time_str}:{text}'"

        if case_name not in case_dict:
            case_dict[case_name] = []
        
        case_dict[case_name].append((time_str, img_info))

    # Generate new JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        json_data = []
        for case_name, img_infos in case_dict.items():
            # Sort by time
            img_infos.sort(key=lambda x: x[0])
            # Build input string
            input_str = '\n'.join(info[1] for info in img_infos)
            new_entry = {
                "instruction": "Generate a structured medical report summarizing endoscopy findings by anatomical site. Follow these rules: 1. Summarize a structure as 'Normal' if all descriptions about it are normal; otherwise, summarize only the abnormality of this structure. 2. If a structure has multiple unique abnormalities, combine them concisely. 3. Unmentioned structures are assumed normal.", 
                "input": input_str,  
                "output": "",  
                "case_name": case_name
            }
            json_data.append(new_entry)
        
        json.dump(json_data, f, ensure_ascii=False, indent=4)

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Generate structured medical report data from jsonl files')
    parser.add_argument('--test_img_file', type=str, 
                       default='../LLaVA/ce-imgs/test_img_to_llava.jsonl',
                       help='Path to test image jsonl file')
    parser.add_argument('--answer_file', type=str,
                       default='../LLaVA/ce-imgs/answer.jsonl',
                       help='Path to answer jsonl file')
    parser.add_argument('--output_file', type=str,
                       default='llava_caption_sum.json',
                       help='Path to output json file')
    
    args = parser.parse_args()
    
    generate_new_jsonl(args.test_img_file, args.answer_file, args.output_file)

if __name__ == '__main__':
    main()