import os
import json
from datetime import datetime
import shutil
# Configuration paths
frame_vis_path = "extracted_keyframe"
duration_dir = '../vce_data/duration'

# Initialize results list
results = []

# Traverse patient folders in frame_vis directory
question_id = 0
for case_folder in os.listdir(frame_vis_path):
    case_path = os.path.join(frame_vis_path, case_folder)
    if os.path.isdir(case_path):
        # Construct duration file path
        duration_file = os.path.join(duration_dir, f"{case_folder}_duration.txt")
        
        # Check if duration file exists
        if not os.path.exists(duration_file):
            print(f"Warning: Duration file not found for {case_folder}, skipping...")
            continue
            
        try:
            # Read video duration
            with open(duration_file, 'r') as f:
                duration_str = f.read().strip()
                duration = datetime.strptime(duration_str, "%H-%M-%S")
                total_seconds = duration.hour * 3600 + duration.minute * 60 + duration.second
        except Exception as e:
            print(f"Error reading duration for {case_folder}: {e}")
            continue
        
        # Check if total_seconds is zero to avoid division by zero
        if total_seconds == 0:
            print(f"Warning: Zero duration for {case_folder}, skipping...")
            continue
            
        for img_name in os.listdir(case_path):
            if "pre" in img_name:
                try:
                    # Extract timestamp from image filename
                    time_str = img_name.split("_")[2].replace(".jpg", "")
                    img_time = datetime.strptime(time_str, "%H-%M-%S")
                    img_seconds = img_time.hour * 3600 + img_time.minute * 60 + img_time.second
                    
                    # Calculate appearance percentage
                    percentage = (img_seconds / total_seconds) * 100
                    
                    # Build result dictionary
                    result = {
                        "question_id": question_id,
                        "image": f"ce-imgs/{case_folder}/{img_name}",
                        "text": f"This image appears in {percentage:.2f}% of the video. Render a clear and concise summary of the image.",
                        "gpt4_answer": None
                    }
                    results.append(result)
                    question_id += 1
                except Exception as e:
                    print(f"Error processing image {img_name} in {case_folder}: {e}")
                    continue

# Write results to JSONL file
output_file = "extracted_keyframe/test_img_to_llava.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure output directory exists

with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"JSONL file generated: {output_file}")
print(f"Total entries: {len(results)}")

shutil.copytree("extracted_keyframe", "../LLaVA/ce-imgs", dirs_exist_ok=True)