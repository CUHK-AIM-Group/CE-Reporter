import os
import torch
import numpy as np
import json

class TestFeatureLoader:
    def __init__(self, video_feature_path, cluster_path, test_case_list, duration=64):
        """
        Dataset class for loading and processing test video features.
        
        Args:
            video_feature_path (str): Path to the directory containing .npy video features.
            cluster_path (str): Path to the directory containing _clusters.json files. 
            test_case_list (list): List of video IDs for testing.
            duration (int): Duration (in frames) of each video segment.
        """
        self.video_feature_path = video_feature_path
        self.cluster_path = cluster_path
        self.duration = duration
        self.case_list = np.load(test_case_list, allow_pickle=True).tolist()

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        """
        Load and segment video features into fixed-length clips.

        Returns:
            dict: Dictionary containing segmented video features and video ID.
        """
        vid = self.case_list[idx]

        # Load video feature
        path = os.path.join(self.video_feature_path, f"{vid}.npy")
        cluster_path = os.path.join(self.cluster_path, f"{vid}_clusters.json")
        with open(cluster_path, 'r') as f:
            clusters = json.load(f)
        representative_indices = [cluster['representative_index'] for cluster in clusters.values()]

        array = np.load(path)[representative_indices]
        feature = torch.from_numpy(array)
        vlen = feature.size(0)
        
        feature_dim = feature.size(1)

        # Generate start and end timestamps
        start_timestamp = []
        end_timestamp = []
        for i in range(vlen // self.duration + 1):
            start_timestamp.append(i * self.duration)
            if i == vlen // self.duration:
                end_timestamp.append(vlen)
            else:
                end_timestamp.append((i + 1) * self.duration)

        # Segment video into clips
        VFeat = []
        for i in range(len(start_timestamp)):
            video_feature = self._get_video_feature(feature, vid, start_timestamp[i], end_timestamp[i], feature_dim)
            VFeat.append(video_feature)

        return {
            'video': VFeat,
            'vid': vid
        }

    def _get_video_feature(self, feature, vid, start, end, feature_dim):
        """
        Extract a fixed-length segment from the full video feature.

        Args:
            feature (torch.Tensor): Full video feature tensor.
            vid (str): Video ID (for debugging).
            start (int): Start frame index.
            end (int): End frame index.
            feature_dim (int): Dimension of feature (feature.shape[1]).

        Returns:
            torch.Tensor: Trimmed or padded video clip of shape (duration, feature_dim).
        """
        feature_cut = torch.zeros((self.duration, feature_dim))
        actual_length = end - start
        feature_cut[:actual_length, :] = feature[start:end, :]

        # Debugging check
        if actual_length <= 0:
            print(f"[Warning] Video {vid} has invalid segment: {start}-{end} with shape {feature.shape}")

        return feature_cut.float()