import argparse
import os
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basic settings
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset', default='MICS-CE', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    
    # Data settings
    parser.add_argument('--seq_len', default=5*60*4, type=int, help='Sequence length')
    parser.add_argument('--batch_size', default=1, type=int)
    
    # Training hyperparameters
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--wd', default=2e-6, type=float, help='Weight decay')
    parser.add_argument('--clip_grad', default=0.0, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--backprop_freq', default=1, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    
    # Loss weights
    parser.add_argument('--weight_keyframelong', type=float, default=0.3,
                        help='Weight for the long-term keyframe prediction loss.')
    parser.add_argument('--weight_keyframeshort', type=float, default=0.3,
                        help='Weight for the short keyframe prediction loss.')
    parser.add_argument('--weight_seg', type=float, default=0.4,
                        help='Weight for the temporal segmentation loss.')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None,
                        help='List of class weights for the segmentation loss. If None, no weighting is applied.')

    # Checkpoint and resume
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--pretrain', default='', type=str,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--checkpoint', default='checkpoints', type=str,
                        help='Base directory to save checkpoints')
    parser.add_argument('--prefix', default='', type=str,
                        help='Prefix for experiment directory name')

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

    # Data paths (required for training)
    parser.add_argument('--video_feature_path', type=str, default='../vce_data/swin_fea', help='Path to video features')
    parser.add_argument('--annotation_path', type=str, default='../vce_data/KeyFrames_label', help='Path to annotation files')
    parser.add_argument('--cluster_path', type=str, default='../vce_data/clusters', help='Path to cluster info')
    parser.add_argument('--train_case_list', type=str, default='../vce_data/npy_files/train_list.npy', help='Path to .npy file containing list of training case IDs')
    parser.add_argument('--valid_case_list', type=str, default='../vce_data/npy_files/val_list.npy', help='Path to .npy file containing list of validation case IDs')

    args = parser.parse_args()
    return args


def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string

    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = (f"{args.prefix}{dt_string}_"
            f"{args.dataset}_len{args.seq_len}_"
            f"pos-{args.pos_enc}_"
            f"bs{args.batch_size}_lr{args.lr}")

    exp_path = os.path.join(args.checkpoint, exp_path)
    log_path = os.path.join(exp_path, 'log')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)

    with open(f'{log_path}/running_command.txt', 'a') as f:
        json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        f.write('\n')

    return log_path, model_path, exp_path