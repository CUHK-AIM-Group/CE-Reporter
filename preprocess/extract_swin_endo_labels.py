# extract_swin_endo_labels.py

import os
import json
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from transformers import SwinConfig, SwinModel
import time
import argparse
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from huggingface_hub import snapshot_download
import torch.nn.functional as F

def parse_args():
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description='Extract Swin and optionally EndoViT features, and generate labels from endoscopic images.')
    
    # Data and Output Paths
    parser.add_argument('--frame-folder', default='../vce_data/frames',
                        help='Path to the folder containing frame images.')
    parser.add_argument('--annotation-folder', default='../vce_data/annotation',
                        help='Path to the folder containing annotation JSON files.')
    parser.add_argument('--swin-output', default='../vce_data/swin_fea',
                        help='Path to save extracted Swin features.')
    parser.add_argument('--endo-output', default='../vce_data/endo_fea',
                        help='Path to save extracted EndoViT features.')
    parser.add_argument('--labels-output', default='../vce_data/KeyFrames_label',
                        help='Path to save generated labels.')
    parser.add_argument('--selectid-output', default='../vce_data/select_ids',
                        help='Path to save selected frame IDs.')
    
    # Model and Processing Parameters
    parser.add_argument('--output-dim', type=int, default=256,
                        help='Output dimension for Swin feature extraction.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for data loading.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for data loading.')
    parser.add_argument('--cuda-device', default='0',
                        help='CUDA device ID(s) to use (e.g., "0" or "0,1").')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Size to resize images (width and height) for Swin/Endo.')
    
    # Optional Features
    parser.add_argument('--enable-endo', action='store_false', default=True,
                        help='Enable extraction of EndoViT features.')
    parser.add_argument('--skip-annotations', action='store_true', default=False,
                        help='If set, skip generating label .npy files and only extract features.')
    
    # Patient Selection (Optional)
    parser.add_argument('--patient-list-file', type=str,
                        help='Path to a text file containing patient IDs (one per line) to process. If not provided, all patients in frame-folder are processed.')

    return parser.parse_args()

# --- Model Definitions ---

class SwinExtractor(nn.Module):
    """Feature extraction model using Swin Transformer."""
    def __init__(self, output_dim=256):
        super(SwinExtractor, self).__init__()
        config = SwinConfig.from_pretrained('microsoft/swin-base-patch4-window7-224-in22k')
        self.model = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224-in22k', config=config)
        self.pool = nn.AdaptiveAvgPool1d(output_dim)

    def forward(self, images):
        features = self.model(images).last_hidden_state
        features_mean = features.mean(dim=1)
        pooled_features = self.pool(features_mean)
        return pooled_features

def load_endovit_from_huggingface(repo_id, model_filename):
    """Load EndoViT model from Hugging Face."""
    model_path = snapshot_download(repo_id=repo_id, revision="main")
    model_weights_path = Path(model_path) / model_filename
    model_weights = torch.load(model_weights_path)['model']
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).eval()
    loading_result = model.load_state_dict(model_weights, strict=False)
    print(f"EndoViT loading result (missing/unexpected): {loading_result}")
    return model

# --- Data Preprocessing and Dataset ---

# Transforms for Swin (ImageNet normalization)
transform_swin = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transforms for EndoViT (EndoViT-specific normalization)
transform_endo = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3464, 0.2280, 0.2228], std=[0.2520, 0.2128, 0.2093])
])

def process_single_image_swin_endo(image_path):
    """Process a single image for both Swin and EndoViT."""
    image = Image.open(image_path).convert('RGB')
    img_swin = transform_swin(image)
    img_endo = transform_endo(image)
    return img_swin, img_endo

class FrameDatasetSwinEndo(Dataset):
    """Dataset for Swin and EndoViT features."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_swin, img_endo = process_single_image_swin_endo(img_path)
        return img_swin, img_endo

# --- Feature Extraction Logic ---

def extract_features_by_time(frame_path, case_name, args):
    """Main function to extract Swin (and optionally EndoViT) features."""
    print(f"--- Extracting features for {case_name} ---")
    start_case_time = time.time()

    # --- 1. Select Middle Frames ---
    image_files = [f for f in os.listdir(frame_path) if f.endswith('.jpg')]
    if not image_files:
        print(f"No frames found in {frame_path}")
        return None, None, None

    image_files.sort()
    images_by_time = defaultdict(list)
    for img_file in image_files:
        parts = img_file.split('_')
        if len(parts) >= 3:
            time_ = parts[2].split('.')[0] 
            images_by_time[time_].append(os.path.join(frame_path, img_file))
        else:
            print(f"Warning: Skipping file with unexpected name format: {img_file}")

    if not images_by_time:
        print(f"No valid time-based frames found in {frame_path}")
        return None, None, None

    selected_images = []
    image_times = []
    selected_ids = []

    for time_, img_files in sorted(images_by_time.items()):
        mid_img_file = img_files[len(img_files) // 2]
        selected_images.append(mid_img_file)
        try:
            selected_ids.append(os.path.basename(mid_img_file).split('_')[1])
        except IndexError:
            selected_ids.append("")
        image_times.append(time_)

    # Save selected IDs
    os.makedirs(args.selectid_output, exist_ok=True)
    np.save(os.path.join(args.selectid_output, f"{case_name}_selectid.npy"), selected_ids)
    print(f"Selected {len(selected_images)} middle frames for {case_name}.")

    # --- 2. Extract Features (Swin and EndoViT in one pass) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Swin model
    print(f"Loading Swin model for {case_name}...")
    swin_model = SwinExtractor(output_dim=args.output_dim)
    swin_model = swin_model.to(device).eval()

    # Load EndoViT model (if enabled)
    endo_model = None
    if args.enable_endo:
        print(f"Loading EndoViT model for {case_name}...")
        try:
            endo_model = load_endovit_from_huggingface("egeozsoy/EndoViT", "pytorch_model.bin")
            endo_model = endo_model.to(device, torch.float16).eval()
        except Exception as e:
            print(f"Error loading EndoViT model for {case_name}: {e}")
            args.enable_endo = False # Disable if loading fails

    # Create dataset and dataloader
    dataset = FrameDatasetSwinEndo(selected_images)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    swin_features_list = []
    endo_features_list = []

    print(f"Extracting features for {case_name}...")
    extraction_start_time = time.time()
    with torch.no_grad():
        for img_swin_batch, img_endo_batch in dataloader:
            img_swin_batch = img_swin_batch.to(device)

            # Extract Swin features
            swin_features = swin_model(img_swin_batch)
            swin_features_list.append(swin_features.cpu())

            # Extract EndoViT features (if enabled)
            if args.enable_endo and endo_model is not None:
                img_endo_batch = img_endo_batch.to(device, torch.float16)
                endo_features_tokens = endo_model.forward_features(img_endo_batch)
                endo_features_pooled = endo_features_tokens.mean(dim=1)
                endo_features_pooled = F.normalize(endo_features_pooled, p=2, dim=1)
                endo_features_list.append(endo_features_pooled)

    print(f"Feature extraction took {time.time() - extraction_start_time:.2f} seconds.")

    # --- 3. Save Swin Features ---
    swin_features_array = None
    if swin_features_list:
        swin_features_array = torch.cat(swin_features_list, dim=0).numpy()
        os.makedirs(args.swin_output, exist_ok=True)
        np.save(os.path.join(args.swin_output, f"{case_name}.npy"), swin_features_array)
        print(f"Saved Swin features shape: {swin_features_array.shape}")
    else:
        print(f"Failed to extract Swin features for {case_name}.")
    del swin_model, swin_features_list, dataset, dataloader
    torch.cuda.empty_cache()

    # --- 4. Save EndoViT Features (if enabled) ---
    endo_features_array = None
    if args.enable_endo and endo_features_list:
        endo_features_array = torch.cat(endo_features_list, dim=0).detach().cpu().numpy()
        os.makedirs(args.endo_output, exist_ok=True)
        np.save(os.path.join(args.endo_output, f"{case_name}.npy"), endo_features_array)
        print(f"Saved EndoViT features shape: {endo_features_array.shape}")
    elif args.enable_endo:
        print(f"Failed to extract EndoViT features for {case_name}.")
    
    if endo_model is not None:
        del endo_model
    if endo_features_list: # Clear list even if not saved
        del endo_features_list
    torch.cuda.empty_cache()

    print(f"--- Completed feature extraction for {case_name} in {time.time() - start_case_time:.2f} seconds ---")
    return swin_features_array, endo_features_array, image_times

# --- Label Generation Logic (from pasted code 1) ---

def time_to_seconds(time_str, delimiter=':'):
    """Convert time string (hh:mm:ss) to seconds."""
    h, m, s = map(int, time_str.split(delimiter))
    return h * 3600 + m * 60 + s

def generate_labels(annotation_folder, imgs_times, case_name, labels_output):
    """Generate binary labels for keyframes (disease presence)."""
    annotation_file = os.path.join(annotation_folder, f"{case_name}.json")
    if not os.path.exists(annotation_file):
        print(f"Annotation file for {case_name} not found.")
        return None

    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)

    keyframes_times = set()
    for kf in annotation_data.get('Keyframe', []):
        if kf['keyframe symbol']:
            keyframes_times.add(kf['time'].replace(':', '-'))

    labels = []
    for time_ in imgs_times:
        if time_ in keyframes_times:
            labels.append(1)
        else:
            labels.append(0)

    labels_array = np.array(labels)
    os.makedirs(labels_output, exist_ok=True)
    np.save(os.path.join(labels_output, f"{case_name}.npy"), labels_array)
    return labels_array

def generate_specific_labels(annotation_folder, imgs_times, case_name, labels_output):
    """Generate specific type labels for keyframes based on matching times."""
    # Attempt to find the annotation file
    annotation_file = os.path.join(annotation_folder, f"{case_name}.json")

    if not os.path.exists(annotation_file):
        print(f"Annotation file for {case_name} not found for specific labels.")
        return None

    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)

    keyframes_times = []
    keyframe_types = {}
    for kf in annotation_data.get('Keyframe', []):
        if kf.get('keyframe symbol') == 1:
            time_str = kf['time'].replace(':', '-')
            keyframes_times.append(time_str)
            keyframe_types[time_str] = int(kf['keyframe type'])

    labels = []
    for time_ in imgs_times:
        time_str = str(time_).replace(':', '-')  # Ensure consistent time format
        if time_str in keyframes_times:
            labels.append(keyframe_types[time_str])
        else:
            labels.append(0)

    labels_array = np.array(labels)
    os.makedirs(labels_output, exist_ok=True)
    np.save(os.path.join(labels_output, f"{case_name}_specific_type.npy"), labels_array)
    return labels_array

def generate_segment_labels(annotation_folder, imgs_times, case_name, labels_output):
    """Generate segment labels for anatomical regions."""
    annotation_file = os.path.join(annotation_folder, f"{case_name}.json")
    if not os.path.exists(annotation_file):
        print(f"Annotation file for {case_name} not found.")
        return None

    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)

    segments = annotation_data.get('Segment', [])
    if not segments:
        print(f"No segments found in annotation file for {case_name}.")
        return None

    segment_labels = {
        "esophagus": 1,
        "stomach": 2,
        "small intestine": 3,
        "large intestine": 4
    }

    labels = []
    for time_ in imgs_times:
        frame_time_in_seconds = time_to_seconds(time_, '-')
        label = 0  # Default label for unknown segments

        for i, segment in enumerate(segments):
            start_time = time_to_seconds(segment['start_time'])
            end_time = time_to_seconds(segment['end_time'])
            if i == len(segments) - 1:
                if frame_time_in_seconds >= start_time:
                    label = segment_labels.get(segment['summary'], 0)
                    break
            else:
                if start_time <= frame_time_in_seconds <= end_time:
                    label = segment_labels.get(segment['summary'], 0)
                    break
        labels.append(label)

    labels_array = np.array(labels)
    os.makedirs(labels_output, exist_ok=True)
    np.save(os.path.join(labels_output, f"{case_name}_segment_labels.npy"), labels_array)
    return labels_array

# --- Main Processing Logic ---

def load_patient_ids(patient_list_file):
    """Load patient IDs from a text file."""
    if not patient_list_file or not os.path.isfile(patient_list_file):
        return None
    try:
        with open(patient_list_file, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(patient_ids)} patient IDs from {patient_list_file}")
        return set(patient_ids)
    except Exception as e:
        print(f"Error reading patient list file {patient_list_file}: {e}")
        return None

def process_patient(frame_path, case_name, args):
    """Process a single patient's data."""
    print(f"\n--- Processing patient: {case_name} ---")
    start_time = time.time()
    
    # Extract features
    swin_fea, endo_fea, img_times = extract_features_by_time(frame_path, case_name, args)
    
    # Generate labels if features were extracted and annotations are not skipped
    if swin_fea is not None and img_times is not None and not args.skip_annotations:
        print(f"Generating labels for {case_name}...")
        try:
            generate_labels(args.annotation_folder, img_times, case_name, args.labels_output)
            generate_specific_labels(args.annotation_folder, img_times, case_name, args.labels_output)
            generate_segment_labels(args.annotation_folder, img_times, case_name, args.labels_output)
            print(f"Labels generated for {case_name}.")
        except Exception as e:
             print(f"Error generating labels for {case_name}: {e}")
             import traceback
             traceback.print_exc() # Print stack trace for debugging
    elif args.skip_annotations:
        print(f"Skipped label generation for {case_name} as requested.")
    else:
         print(f"Skipped label generation for {case_name} due to missing Swin features.")

    print(f"--- Completed processing {case_name} in {time.time() - start_time:.2f} seconds ---")

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print("CUDA_VISIBLE_DEVICES set to:", args.cuda_device)

    # Create necessary output directories
    os.makedirs(args.swin_output, exist_ok=True)
    if args.enable_endo:
        os.makedirs(args.endo_output, exist_ok=True)
    if not args.skip_annotations:
        os.makedirs(args.labels_output, exist_ok=True)
    os.makedirs(args.selectid_output, exist_ok=True)

    # Load list of patients to process
    patient_ids_to_process = load_patient_ids(args.patient_list_file)

    # Get list of patient directories
    if not os.path.isdir(args.frame_folder):
        raise FileNotFoundError(f"Frame folder does not exist: {args.frame_folder}")

    case_dirs = [
        d for d in os.listdir(args.frame_folder)
        if os.path.isdir(os.path.join(args.frame_folder, d))
    ]

    if not case_dirs:
        print(f"No patient directories found in {args.frame_folder}")
        return

    print(f"Found {len(case_dirs)} potential patient directories.")

    # Filter case_dirs if a patient list is provided
    if patient_ids_to_process is not None:
        case_dirs = [d for d in case_dirs if d in patient_ids_to_process]
        print(f"Filtered to {len(case_dirs)} patient directories based on the list.")

    if not case_dirs:
        print("No patient directories to process after filtering.")
        return

    print(f"Starting to process {len(case_dirs)} patients...")
    start_total_time = time.time()

    for i, case_name in enumerate(case_dirs):
        frame_path = os.path.join(args.frame_folder, case_name)
        if not os.path.isdir(frame_path):
            print(f"Warning: Expected directory not found, skipping: {frame_path}")
            continue

        print(f"\n--- Processing patient {i+1}/{len(case_dirs)}: {case_name} ---")
        try:
            process_patient(frame_path, case_name, args)
        except Exception as e:
            print(f"Error processing patient {case_name}: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging

    print(f"\n--- All processing completed in {time.time() - start_total_time:.2f} seconds ---")

if __name__ == "__main__":
    main()