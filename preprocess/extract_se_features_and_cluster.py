# extract_se_cluster.py
import os
import json
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import time
import argparse
from tqdm import tqdm
import torch.nn.functional as F

def parse_args():
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description='Extract SEResNet features and perform clustering to generate cluster.json files.')
    # Data and Output Paths
    parser.add_argument('--frame-folder', default='../vce_data/frames',
                        help='Path to the folder containing frame images.')
    parser.add_argument('--se-output', default='../vce_data/seresnet_fea',
                        help='Path to save extracted SEResNet features.')
    parser.add_argument('--cluster-output', default='../vce_data/clusters',
                        help='Path to save generated cluster JSON files.')
    # Model and Processing Parameters
    parser.add_argument('--se-model-path', type=str, default='../vce_data/pretrained_weight/seresnet.pth',
                        help='Path to the trained SEResNet model weights file (e.g., model_best.pth).')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for data loading.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for data loading.')
    parser.add_argument('--cuda-device', default='0',
                        help='CUDA device ID(s) to use (e.g., "0" or "0,1").')
    parser.add_argument('--image-size', type=int, default=320,
                        help='Size to resize images (width and height) for SEResNet.')
    parser.add_argument('--similarity-threshold', type=float, default=0.94,
                        help='Cosine similarity threshold for clustering.')
    # Patient Selection (Optional)
    parser.add_argument('--patient-list-file', type=str,
                        help='Path to a text file containing patient IDs (one per line) to process. If not provided, all patients in frame-folder are processed.')
    # --- Keyframe Integration ---
    parser.add_argument('--annotation-folder', type=str, default="./vce_data/annotation", 
                        help='Path to the folder containing annotation JSON files. If provided, enables updating clusters with representative_key_index from keyframes.')
    parser.add_argument('--keyframe-key', type=str, default="Keyframe",
                        help='The key in the annotation JSON file that contains the list of keyframes. Defaults to "Keyframe".')
    # ---------------------------------------------
    return parser.parse_args()

# --- Model Definitions ---
class SEResNetExtractor(nn.Module):
    """Feature extraction model using SE-ResNet50."""
    def __init__(self, model_path): # Assuming 18 classes based on second code
        super(SEResNetExtractor, self).__init__()
        # Load pre-trained SE-ResNet50
        model = timm.create_model('seresnet50', pretrained=True)
        self.fE = nn.Sequential(*list(model.children())[:-1])
        # Load custom weights
        self._load_weights(model_path)

    def _load_weights(self, model_path):
        """Load custom weights, handling potential 'module.' prefix."""
        state_dict = torch.load(model_path)
        self.fE.load_state_dict(state_dict)

    def forward(self, x):
        """Forward pass, returns features and logits."""
        features = self.fE(x)
        features_normalized = F.normalize(features, p=2, dim=1)
        return features_normalized

# --- Data Preprocessing and Dataset ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
transform_se = A.Compose([
        A.Resize(320, 320), 
        A.Normalize(mean=(140.6/255, 93.1/255, 60.5/255), std=(74.1/255, 57.1/255, 35.4/255)),  
        ToTensorV2()  
    ])

def process_single_image_se(image_path):
    """Process a single image for SE-ResNet."""
    image2 = cv2.imread(image_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB).astype(float)
    img_se = transform_se(image=image2)['image']
    return img_se

class FrameDatasetSE(Dataset):
    """Dataset for SE-ResNet features."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_se = process_single_image_se(img_path)
        return img_se

# --- Clustering Logic (adapted from second code) ---
def cluster_frames(features, image_times, image_names, similarity_threshold=0.94):
    """
    Perform redundancy reduction clustering based on average cluster features.
    Optimized for representativeness and redundancy removal.
    Args:
        features (np.ndarray): Frame features array, shape (N, feature_dim).
        image_times (list): List of frame times.
        image_names (list): List of frame filenames.
        similarity_threshold (float): Cosine similarity threshold.
    Returns:
        dict: Clustering result, key is representative frame name, value is dict with time_range and frames.
    """
    # Handle empty features
    if len(features) == 0:
        return {}
    # Move features to GPU for faster computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_tensor = torch.tensor(features, device=device, dtype=torch.float32)
    num_frames = len(features_tensor)
    # Initialize the first cluster
    clusters = []
    current_cluster = {
        "start_idx": 0,
        "start_time": image_times[0],
        "end_time": image_times[0],
        "frames": [image_names[0]],
        "mean_feature": features_tensor[0].clone()  # Initial average feature
    }
    # Process frames sequentially (starting from the second frame)
    for i in tqdm(range(1, num_frames), desc="Clustering"):
        # Calculate similarity to the current cluster's average feature
        similarity = torch.nn.functional.cosine_similarity(
            current_cluster["mean_feature"].unsqueeze(0),
            features_tensor[i].unsqueeze(0)
        ).item()
        # If similar enough, add to the current cluster and update average
        if similarity >= similarity_threshold:
            current_cluster["end_time"] = image_times[i]
            current_cluster["frames"].append(image_names[i])
            # Incrementally update the average feature
            n = len(current_cluster["frames"])
            current_cluster["mean_feature"] = (
                current_cluster["mean_feature"] * (n - 1) + features_tensor[i]
            ) / n
        else:
            # If not similar, finalize the current cluster and start a new one
            clusters.append(current_cluster)
            current_cluster = {
                "start_idx": i,
                "start_time": image_times[i],
                "end_time": image_times[i],
                "frames": [image_names[i]],
                "mean_feature": features_tensor[i].clone()
            }
    # Add the last cluster
    clusters.append(current_cluster)
    # Format output: Select the frame closest to the cluster's average as the representative
    result = defaultdict(lambda: {"time_range": [], "frames": []})
    for cluster in clusters:
        # Get features for frames in this cluster
        cluster_features = features_tensor[
            cluster["start_idx"]:cluster["start_idx"] + len(cluster["frames"])
        ]
        # Calculate similarities to the cluster's mean
        similarities = torch.nn.functional.cosine_similarity(
            cluster_features, cluster["mean_feature"].unsqueeze(0)
        )
        # Find the index of the most representative frame
        rep_idx = torch.argmax(similarities).item()
        rep_name = cluster["frames"][rep_idx]
        # Store the result
        result[rep_name]["time_range"] = [cluster["start_time"], cluster["end_time"]]
        result[rep_name]["frames"] = cluster["frames"]
    # Clean up GPU memory
    del features_tensor
    return dict(result)

def update_cluster_keys(clusters, feature_indices):
    """
    Update cluster keys to include the representative frame's index in the original feature array.
    Args:
        clusters (dict): The clustering result dictionary.
        feature_indices (dict): Mapping from image name to its index in the full feature array.
    Returns:
        dict: Updated clusters dictionary with 'representative_index'.
    """
    updated_clusters = {}
    for cluster_key, cluster_data in clusters.items():
        # Add the index of the representative frame (key) in the original features
        cluster_data["representative_index"] = feature_indices[cluster_key]
        updated_clusters[cluster_key] = cluster_data
    return updated_clusters

# --- New Functions for Keyframe Integration ---
def load_keyframes(annotation_folder, patient_name, keyframe_key="Keyframe"):
    """
    Load keyframe time information for a patient.

    Args:
        annotation_folder (str): Path to the annotation folder.
        patient_name (str): Name/ID of the patient.
        keyframe_key (str): The key in the JSON file containing the keyframe list.

    Returns:
        tuple:
            set: A set of keyframe times formatted as "hh-mm-ss".
            dict: A dictionary mapping keyframe times ("hh-mm-ss") to their original JSON data.
                  This is useful if you need more details than just the time later.
    """
    annotation_path = os.path.join(annotation_folder, f"{patient_name}.json")
    keyframe_times = set()
    keyframe_details = {}

    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file not found for {patient_name}: {annotation_path}")
        return keyframe_times, keyframe_details

    try:
        with open(annotation_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for {patient_name}: {e}")
        return keyframe_times, keyframe_details
    except Exception as e:
        print(f"Error reading annotation file for {patient_name}: {e}")
        return keyframe_times, keyframe_details

    keyframes_list = data.get(keyframe_key, [])
    for kf in keyframes_list:
        # Assuming the keyframe dict has a "time" field like "12:34:56"
        time_str = kf.get("time")
        if time_str:
            # Convert "hh:mm:ss" to "hh-mm-ss" to match frame naming
            formatted_time = time_str.replace(":", "-")
            keyframe_times.add(formatted_time)
            # Store the full keyframe data for potential future use
            keyframe_details[formatted_time] = kf
        else:
            print(f"Warning: Keyframe entry missing 'time' field in {annotation_path}: {kf}")

    return keyframe_times, keyframe_details

def update_cluster_keys_renew(clusters, keyframe_times, feature_indices):
    """
    Update clusters by adding 'representative_key_index' for clusters containing keyframes.

    This function iterates through clusters. For each cluster, it checks if any of its
    frames corresponds to a known keyframe time. If so, it adds the 'representative_key_index'
    key to the cluster data, using the feature index of that keyframe.

    Args:
        clusters (dict): The original clustering result dictionary.
        keyframe_times (set): A set of keyframe times formatted as "hh-mm-ss".
        feature_indices (dict): Mapping from image name (basename) to its index in the features array.

    Returns:
        dict: Updated clusters dictionary. Clusters containing keyframes will have
              a 'representative_key_index' key added.
    """
    updated_clusters = {}
    for cluster_key, cluster_data in clusters.items():
        frames = cluster_data["frames"]

        # Check each frame in the cluster to see if it's a keyframe
        for frame_name in frames:
            time_in_frame = frame_name.split("_")[-1].split(".")[0]
            if time_in_frame in keyframe_times:
                # Found a keyframe in this cluster
                keyframe_index = feature_indices.get(frame_name)
                if keyframe_index is not None:
                    # Add the key and its value (feature index of the keyframe)
                    cluster_data["representative_key_index"] = keyframe_index
                    break
                else:
                    print(f"Warning: Keyframe {frame_name} found in cluster but not in feature_indices.")

        # Add the (potentially updated) cluster data to the new dictionary
        updated_clusters[cluster_key] = cluster_data

    return updated_clusters


# --- Main Feature Extraction and Clustering Logic ---
def extract_se_features_and_cluster(frame_path, case_name, args):
    print(f"--- Extracting SE features and clustering for {case_name} ---")
    # --- 1. Select Middle Frames ---
    start_case_time = time.time()
    image_files = [f for f in os.listdir(frame_path) if f.endswith('.jpg')]
    if not image_files:
        print(f"No frames found in {frame_path}")
        return None, None, None
    image_files.sort()
    images_by_time = defaultdict(list)
    for img_file in image_files:
        parts = img_file.split('_')
        if len(parts) >= 3:
            time_ = parts[2].split('.')[0] # Assumes format like frame_000001_12-34-56.jpg
            images_by_time[time_].append(os.path.join(frame_path, img_file))
        else:
            print(f"Warning: Skipping file with unexpected name format: {img_file}")
    if not images_by_time:
        print(f"No valid time-based frames found in {frame_path}")
        return None, None, None
    selected_images = []
    image_times = []
    image_names = []
    for time_, img_files in sorted(images_by_time.items()):
        mid_img_file = img_files[len(img_files) // 2]
        selected_images.append(mid_img_file)
        image_names.append(os.path.basename(mid_img_file))
        image_times.append(time_)
    # --- 2. Extract SE Features ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading SE-ResNet model from {args.se_model_path} for {case_name}...")
    try:
        se_model = SEResNetExtractor(model_path=args.se_model_path)
        se_model = se_model.to(device).eval()
    except Exception as e:
        print(f"Error loading SE-ResNet model for {case_name}: {e}")
        return None
    # Create dataset and dataloader
    dataset = FrameDatasetSE(selected_images)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    se_features_list = []
    print(f"Extracting SE features for {case_name}...")
    extraction_start_time = time.time()
    with torch.no_grad():
        for img_se_batch in dataloader:
            img_se_batch = img_se_batch.to(device)
            se_features = se_model(img_se_batch)
            se_features_list.append(se_features)
    print(f"SE feature extraction took {time.time() - extraction_start_time:.2f} seconds.")
    se_features_array = None
    if se_features_list:
        se_features_array = torch.cat(se_features_list, dim=0).detach().cpu().numpy()
        os.makedirs(args.se_output, exist_ok=True)
        se_npy_path = os.path.join(args.se_output, f"{case_name}.npy")
        np.save(se_npy_path, se_features_array)
        print(f"Extracted SE features shape: {se_features_array.shape}")
    else:
        print(f"Failed to extract SE features for {case_name}.")
        return None
    del se_model, se_features_list, dataset, dataloader
    torch.cuda.empty_cache()

    # --- 3. Perform Clustering ---
    print(f"Clustering SE features for {case_name}...")
    clustering_start_time = time.time()
    feature_indices = {img_name: idx for idx, img_name in enumerate(image_names)}
    clusters = cluster_frames(se_features_array, image_times, image_names, args.similarity_threshold)
    clusters = update_cluster_keys(clusters, feature_indices)

    # --- 4. Optional Keyframe Integration ---
    if args.annotation_folder:
        print(f"Loading keyframes for {case_name} from {args.annotation_folder}...")
        keyframe_times, _ = load_keyframes(args.annotation_folder, case_name, args.keyframe_key)
        if keyframe_times:
            print(f"Found {len(keyframe_times)} keyframes for {case_name}. Updating clusters...")
            # Update clusters with representative_key_index where applicable
            clusters = update_cluster_keys_renew(clusters, keyframe_times, feature_indices)
        else:
            print(f"No keyframes found or loaded for {case_name}. Clusters not updated with keyframe indices.")

    # Save the (potentially updated) cluster JSON file
    os.makedirs(args.cluster_output, exist_ok=True)
    cluster_json_path = os.path.join(args.cluster_output, f"{case_name}_clusters.json")
    try:
        with open(cluster_json_path, 'w') as f:
            json.dump(clusters, f, ensure_ascii=False, indent=4)
        print(f"Saved cluster JSON: {cluster_json_path}")
    except Exception as e:
        print(f"Error saving cluster JSON for {case_name}: {e}")
    print(f"--- Completed SE extraction and clustering for {case_name} in {time.time() - start_case_time:.2f} seconds ---")
    return se_features_array # Return full features if needed elsewhere

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
    # Extract SE features and cluster
    se_fea = extract_se_features_and_cluster(frame_path, case_name, args)
    if se_fea is not None:
        print(f"SE features extracted and clustered for {case_name}.")
    else:
        print(f"Failed to process {case_name}.")
    print(f"--- Completed processing {case_name} in {time.time() - start_time:.2f} seconds ---")

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print("CUDA_VISIBLE_DEVICES set to:", args.cuda_device)

    # Validate model path
    if not os.path.isfile(args.se_model_path):
        raise FileNotFoundError(f"SE model path does not exist: {args.se_model_path}")
    
    # Create necessary output directories
    os.makedirs(args.se_output, exist_ok=True)
    os.makedirs(args.cluster_output, exist_ok=True)
    
    # Load list of patients to process
    patient_ids_to_process = load_patient_ids(args.patient_list_file)
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
