import argparse
import os
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basic settings
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    
    # Model architecture parameters
    parser.add_argument('--sim', default='cos', type=str)
    parser.add_argument('--pos_enc', default='learned', type=str)
    parser.add_argument('--model_width', default=256, type=int, help='Model dimension')
    parser.add_argument('--model_heads', default=8, type=int, help='Number of attention heads')
    parser.add_argument('--model_layers', default=4, type=int, help='Number of transformer layers') 
    parser.add_argument('--model_text_token_len', default=120, type=int, help='Maximum text token length')
    parser.add_argument('--model_types', default=3, type=int, help='Number of input types')
    parser.add_argument('--model_fusion_layers', default=4, type=int, help='Number of fusion layers')
    parser.add_argument('--model_keyframe_outputdim', default=2, type=int, help='Keyframe output dimension')
    parser.add_argument('--model_structure_outputdim', default=5, type=int, help='Structure output dimension')

    # Test arguments
    # Model paths
    parser.add_argument('--modelpath', type=str, default="model_weight/latest.pth.tar", help='Path to trained model checkpoint')
    parser.add_argument('--video_feature_path', type=str, default='../vce_data/swin_fea', help='Path to video features')
    parser.add_argument('--cluster_path', type=str, default='../vce_data/clusters', help='Path to cluster info')
    parser.add_argument('--test_case_list', type=str, default='../vce_data/npy_files/test_list.npy', help='Path to .npy file containing list of test case IDs')
    parser.add_argument('--senetfea_path', type=str, default='../vce_data/seresnet_fea', help='Path to senet feature')
    parser.add_argument('--endovitfea_path', type=str, default='../vce_data/endo_fea', help='Path to endovit feature')
    parser.add_argument('--cluster_json_path', type=str, default='../vce_data/clusters', help='Path to cluster JSON file')

    # Data settings
    parser.add_argument('--seq_len', default=5*60*4, type=int, help='Sequence length')
    
    # Output settings
    parser.add_argument('--vis_path', type=str, default='extracted_keyframe', help='Path to save visualization results')
    
    # Keyframe selection hyperparameters
    parser.add_argument('--prob_threshold', type=float, default=0.6, help='Probability threshold for keyframe selection')
    parser.add_argument('--sim_threshold_a', type=float, default=0.90, help='Similarity threshold for senet features')
    parser.add_argument('--sim_threshold_b', type=float, default=0.96, help='Similarity threshold for endovit features')


    args = parser.parse_args()
    return args