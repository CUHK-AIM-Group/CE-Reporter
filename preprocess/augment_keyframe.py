import os
import numpy as np
import torch
import json
import argparse

def update_labels_with_loaded_similarity_structure_constrain(
    patient_ids,
    label_dir,
    clusters_dir,
    feature_dir,
    output_dir,
    similarity_threshold=(0.8, 0.9, 0.8, 0.9)
):
    """ 
    Update keyframe labels based on similarity and structure constraints.
    Args:
        patient_ids (list): List of patient IDs.
        label_dir (str): Directory containing original labels.
        clusters_dir (str): Directory containing cluster files.
        feature_dir (str): Directory containing feature files.
        output_dir (str): Directory to save updated labels.
        similarity_threshold (tuple): Similarity thresholds for each structure.
    """
    for patient in patient_ids:
        cluster_file = os.path.join(clusters_dir, f"{patient}_clusters.json")
        label_file = os.path.join(label_dir, f"{patient}_specific_type.npy")
        structure_file = os.path.join(label_dir, f"{patient}_segment_labels.npy")

        # Load data
        labels = np.load(label_file)  # full labels (before deduplication)
        structures = np.load(structure_file)  # structure labels
        features = np.load(os.path.join(feature_dir, f"{patient}.npy"))

        print(f"{patient} *** Original labels: {np.sum(labels>=1)}")

        # Load clusters to get representative indices (deduplication)
        with open(cluster_file, 'r') as f:
            clusters = json.load(f)

        representative_indices = [
            cluster.get('representative_key_index', cluster['representative_index'])
            for cluster in clusters.values()
        ]

        # Deduplicate labels and structures
        labels_dedup = labels[representative_indices]
        structures_dedup = structures[representative_indices]
        features_dedup = features[representative_indices]

        # Compute similarity matrix
        features_torch = torch.from_numpy(features_dedup).cuda()
        similarity_matrix = torch.matmul(features_torch, features_torch.T).cpu().numpy()

        # Initialize updated labels
        updated_labels = np.zeros_like(labels_dedup)

        stats = []

        for i, (sim_row, structure) in enumerate(zip(similarity_matrix, structures_dedup)):
            current_label = labels_dedup[i]
            threshold = None

            # Set threshold based on label type
            if current_label in [2, 3]:  # esophagus and cardia
                threshold = similarity_threshold[0]
            elif current_label in [1, 2, 3, 4, 5, 6, 7]:
                threshold = similarity_threshold[int(structure) - 1]

            if threshold is not None:
                # Only consider same structure frames
                mask = (sim_row > threshold) & (structures_dedup == structure)

                # Limit search range for label 4 and 7 (pylorus and ileocecal valve)
                if current_label in [4, 7]:
                    start_idx = max(0, i - 50)
                    end_idx = min(len(labels_dedup), i + 51)
                    mask = mask[start_idx:end_idx]
                    updated_labels[start_idx:end_idx][mask] = current_label
                else:
                    updated_labels[mask] = current_label

                stats.append(mask.sum())

        # Convert to binary label (0 or 1)
        updated_labels[updated_labels > 0] = 1

        # Check consistency
        if not np.all(updated_labels[labels_dedup > 0] == 1):
            print(f"⚠️ Warning: Some original labels in patient {patient} are not preserved!")

        print(f"{patient} *** Total updated labels: {np.sum(updated_labels)}")

        # Save updated labels
        save_path = os.path.join(output_dir, f"{patient}_long.npy")
        np.save(save_path, updated_labels)
        print(f"✅ Saved updated labels for {patient} to {save_path}")



def main():
    parser = argparse.ArgumentParser(description="Expand keyframe labels using similarity and structure constraints.")
    parser.add_argument('--patient_list_path', type=str, default="../vce_data/npy_files/train_list.npy", help="Path to patient list (.npy)")
    parser.add_argument('--label_dir', type=str, default="../vce_data/KeyFrames_label", help="Directory of original labels")
    parser.add_argument('--clusters_dir', type=str, default="../vce_data/clusters", help="Directory of cluster files")
    parser.add_argument('--feature_dir', type=str, default="../vce_data/seresnet_fea", help="Directory of feature files")
    parser.add_argument('--output_dir', type=str, default="../vce_data/KeyFrames_label", help="Directory to save updated labels")
    parser.add_argument('--thresholds', nargs=4, type=float, default=[0.8, 0.9, 0.8, 0.9], help="Similarity thresholds for each structure")

    args = parser.parse_args()

    # Load patient list
    patient_list = np.load(args.patient_list_path, allow_pickle=True).tolist()

    # Run label expansion
    update_labels_with_loaded_similarity_structure_constrain(
        patient_ids=patient_list,
        label_dir=args.label_dir,
        clusters_dir=args.clusters_dir,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        similarity_threshold=tuple(args.thresholds)
    )


if __name__ == "__main__":
    main()