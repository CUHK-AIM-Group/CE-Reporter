import os
import torch
import numpy as np
import json

def pad_sequence_by_last(sequences):
    """
    Pad sequences by repeating the last element to match the maximum length.
    
    Args:
        sequences (list): List of tensors with shape (seq_len, ...).
        
    Returns:
        torch.Tensor: Padded tensor with shape (batch_size, max_len, ...).
    """
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, 0.0)
    
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor 
        out_tensor[i, length:, ...] = tensor[-1, ...]
    return out_tensor


class TrainFeatureLoader:
    def __init__(self,
                video_feature_path,
                annotation_path,
                cluster_path,
                mode,
                train_cases_list,
                valid_cases_list,
                duration=64):
        """
        Dataset class for loading and processing training/validation video features.
        
        Args:
            video_feature_path (str): Path to directory containing video .npy files.
            annotation_path (str): Path to directory containing annotation .npy files.
            cluster_path (str): Path to the directory containing _clusters.json files.
            mode (str): 'train' or 'val'.
            train_cases_list (list): List of training video IDs.
            valid_cases_list (list): List of validation video IDs.
            duration (int): Duration of each video segment in frames.
        """
        self.video_feature_path = video_feature_path
        self.annotation_path = annotation_path
        self.cluster_path = cluster_path
        self.duration = duration

        train_vids = np.load(train_cases_list, allow_pickle=True).tolist()
        valid_vids = np.load(valid_cases_list, allow_pickle=True).tolist()

        # Because using clips, try multiple repeated input samples
        if mode == 'train':
            self.case_list = train_vids * 2
        elif mode == 'val':
            self.case_list = valid_vids * 5

    def __len__(self):
        return len(self.case_list)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences.
        """
        out_batch = {}
        out_batch['video'] = pad_sequence_by_last([sample['video'] for sample in batch])
        out_batch['video_label'] = pad_sequence_by_last([sample['video_label'] for sample in batch])
        out_batch['video_label_short'] = pad_sequence_by_last([sample['video_label_short'] for sample in batch])
        out_batch['video_seg_label'] = pad_sequence_by_last([sample['video_seg_label'] for sample in batch])
        out_batch['vid'] = [sample['vid'] for sample in batch]
        return out_batch 

    def __getitem__(self, index):
        """
        Load and segment video features and corresponding labels.
        
        Returns:
            dict: Dictionary containing video features, labels, and video ID.
        """
        vid = self.case_list[index]

        cluster_path = os.path.join(self.cluster_path, f"{vid}_clusters.json")
        with open(cluster_path, 'r') as f:
            clusters = json.load(f)
        
        representative_indices = [cluster.get('representative_key_index', cluster['representative_index']) for cluster in clusters.values()]

        # Load features and labels
        feature_path = os.path.join(self.video_feature_path, f"{vid}.npy")
        label_short_path = os.path.join(self.annotation_path, f"{vid}.npy")
        label_long_path = os.path.join(self.annotation_path, f"{vid}_long.npy")
        segment_label_path = os.path.join(self.annotation_path, f"{vid}_segment_labels.npy")

        feature = torch.from_numpy(np.load(feature_path)[representative_indices])
        video_label_short = torch.from_numpy(np.load(label_short_path)[representative_indices])
        segment_lab = torch.from_numpy(np.load(segment_label_path)[representative_indices])
        video_label = torch.from_numpy(np.load(label_long_path))  ## The long label is augmented based on redundancy

        # Verify alignment
        assert feature.shape[0] == video_label.shape[0] == video_label_short.shape[0] == segment_lab.shape[0]
        vlen = feature.size(0)

        # Randomly sample a segment
        start_timestamp = np.random.randint(0, vlen - self.duration + 1) if vlen - self.duration + 1 >= 1 else 0
        end_timestamp = start_timestamp + self.duration
        
        video_feature, video_label, video_label_short, video_seg_label = self._get_video_feature(
            feature, video_label, video_label_short, segment_lab, vid, start_timestamp, end_timestamp) 

        return {
            'video': video_feature,
            'video_label': video_label,
            'video_label_short': video_label_short,
            'video_seg_label': video_seg_label,
            'vid': vid
        }

    def _get_video_feature(self, feature, video_label, video_label_short, segment_lab, vid, start, end):
        """
        Extract a fixed-length segment from video features and corresponding labels.
        
        Args:
            feature (torch.Tensor): Full video features.
            video_label (torch.Tensor): Long-term labels.
            video_label_short (torch.Tensor): Short-term labels.
            segment_lab (torch.Tensor): Segment labels.
            vid (str): Video ID for debugging.
            start (int): Start frame index.
            end (int): End frame index.
            
        Returns:
            tuple: Trimmed or padded video features and labels.
        """
        feature_dim = feature.size(1) if feature.dim() > 1 else 0
        label_dim = video_label.size(1) if video_label.dim() > 1 else 1
        short_label_dim = video_label_short.size(1) if video_label_short.dim() > 1 else 1
        seg_label_dim = segment_lab.size(1) if segment_lab.dim() > 1 else 1

        if feature.shape[0] > end:
            # Normal case: extract segment
            feature_cut = feature[start:end, :]
            ilab_cut = video_label[start:end]
            ilab_cut_short = video_label_short[start:end]
            slab_cut = segment_lab[start:end]
        else: 
            # If feature length is insufficient, pad by repeating the last element
            feature_cut = feature[start:, :]
            tmp = feature_cut[-1].unsqueeze(0).repeat(self.duration, 1)
            tmp[0:feature_cut.shape[0], :] = feature_cut
            feature_cut = tmp

            ilab_cut = video_label[start:]
            tmp = ilab_cut[-1].unsqueeze(0).repeat(self.duration, 1)
            if label_dim > 1:
                tmp[0:ilab_cut.shape[0], :] = ilab_cut
            else:
                tmp[0:ilab_cut.shape[0]] = ilab_cut
            ilab_cut = tmp

            ilab_cut_short = video_label_short[start:]
            tmp = ilab_cut_short[-1].unsqueeze(0).repeat(self.duration, 1)
            if short_label_dim > 1:
                tmp[0:ilab_cut_short.shape[0], :] = ilab_cut_short
            else:
                tmp[0:ilab_cut_short.shape[0]] = ilab_cut_short
            ilab_cut_short = tmp

            slab_cut = segment_lab[start:]
            tmp = slab_cut[-1].unsqueeze(0).repeat(self.duration, 1)
            if seg_label_dim > 1:
                tmp[0:slab_cut.shape[0], :] = slab_cut
            else:
                tmp[0:slab_cut.shape[0]] = slab_cut
            slab_cut = tmp

        # Debugging check
        if feature_cut.size(0) == 0: 
            print(f'Error log: {vid} with shape {feature.shape} {start}-{end}, is size 0')

        return feature_cut.float(), ilab_cut.long(), ilab_cut_short.long(), slab_cut.long()