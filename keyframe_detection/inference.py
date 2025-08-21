import sys
sys.path.append('model/')
sys.path.append('data/')
import os
import sys
import torch
from torch.utils import data 
import numpy as np 
import random 
from tqdm import tqdm
import torch.cuda.amp as amp 
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from test_config import parse_args
from data.loader_infer import TestFeatureLoader
from model.multi_task_framework import Keyframe_detect
import shutil


def get_dataset(args):
    """Load test dataset."""
    D = TestFeatureLoader
    test_dataset = D(
        video_feature_path=args.video_feature_path,
        cluster_path=args.cluster_path,
        test_case_list=args.test_case_list,
        duration=args.seq_len)

    test_loader = DataLoader(test_dataset,
        batch_size=1, num_workers=args.num_workers, pin_memory=True, drop_last=False, shuffle=False
    )

    return test_dataset, test_loader

def setup(args):
    """Setup CUDA and random seeds."""
    # CUDA setting
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device('cuda')

        num_gpu = len(str(args.gpu).split(','))
        args.num_gpu = num_gpu
        args.batch_size = num_gpu * args.batch_size
        print('=> Effective BatchSize = %d' % args.batch_size)
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Run with CPU')

    # general setting
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    return device

@torch.no_grad()
def inference(args):
    """Main inference function."""
    device = setup(args)

    model = Keyframe_detect(
        sim=args.sim,
        pos_enc=args.pos_enc,
        width=args.model_width,
        heads=args.model_heads,
        layers=args.model_layers,
        in_token=args.seq_len + 10,
        text_token_len=args.model_text_token_len,
        types=args.model_types,
        fusion_layers=args.model_fusion_layers,
        keyframe_outputdim=args.model_keyframe_outputdim,
        structure_outputdim=args.model_structure_outputdim
    )

    model.to(device)
    
    modelpath = args.modelpath
    checkpoint = torch.load(modelpath, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    _, loader = get_dataset(args)
    test(model, loader, device, args)

def Long2short(img_outputs, sim_matrix_a, sim_matrix_b, seg_outputs, keyframe_outputdim, structure_outputdim, prob_threshold, sim_threshold_a, sim_threshold_b, prefix=None):
    """Select keyframes based on similarity and category constraints."""
    # Initialize prediction probabilities
    pred_likel = torch.softmax(img_outputs.reshape(-1, keyframe_outputdim), dim=1)[:sim_matrix_a.shape[0], -1]
    seg_outputs = seg_outputs.reshape(-1, structure_outputdim)[:sim_matrix_a.shape[0], :]

    # Category limits: max frames per category
    category_limits = {1: 1, 2: 3, 3: 10, 4: 2}
    shortlist = {category: [] for category in category_limits}  # Store keyframes for each category

    # Get frame categories
    categories = seg_outputs.argmax(dim=1).data.cpu().numpy()
    
    # Stage 1: Normal selection of high probability frames
    while True:
        # Check if all categories have reached their limits
        if all(len(shortlist[cat]) >= category_limits[cat] for cat in category_limits):
            break

        # Find the frame with maximum probability
        _, pred_idx = pred_likel.sort(descending=True)
        pred_idx = pred_idx[0].item()  # Get index of highest probability frame
        max_prob = pred_likel[pred_idx].item()

        if max_prob < prob_threshold:  # If max probability is below threshold, move to stage 2
            break

        # Current frame category
        current_category = categories[pred_idx]
        
        if current_category == 0:
            pred_likel[pred_idx] = 0
            continue

        # Check category limit
        if len(shortlist[current_category]) >= category_limits.get(current_category, 0):
            pred_likel[pred_idx] = 0  # Skip and zero out
            continue

        # Add current frame to keyframe list
        shortlist[current_category].append(pred_idx)

        # Get similar frames (within same category)
        same_category_idx = np.where(categories == current_category)[0]
        same_category_idx_local = np.where(same_category_idx == pred_idx)[0][0]

        select_sim_a = sim_matrix_a[pred_idx][same_category_idx]
        simi_idx_a = same_category_idx[
            np.nonzero(select_sim_a + np.eye(len(same_category_idx))[same_category_idx_local] > sim_threshold_a)
        ]

        select_sim_b = sim_matrix_b[pred_idx][same_category_idx]
        simi_idx_b = same_category_idx[
            np.nonzero(select_sim_b + np.eye(len(same_category_idx))[same_category_idx_local] > sim_threshold_b)
        ]

        # Remove similar frames
        pred_likel[simi_idx_a] = 0.
        pred_likel[simi_idx_b] = 0.

    # Stage 2: Ensure each category has at least one frame
    for category in category_limits:
        if len(shortlist[category]) == 0:  # If category has no frames
            # Find the frame with highest probability in this category
            category_indices = np.where(categories == category)[0]
            if len(category_indices) > 0:  # Ensure category has frames to choose from
                category_probs = pred_likel[category_indices]
                max_prob_idx = category_indices[category_probs.argmax()]
                shortlist[category].append(max_prob_idx)
                
    # Compile final keyframe list
    final_shortlist = []
    for category in sorted(shortlist.keys()):
        final_shortlist.extend(shortlist[category])

    return final_shortlist

def save_frames(ind_list, vid, vis_path, cluster_json_path):
    """Save selected keyframes to disk."""
    os.makedirs(vis_path, exist_ok=True)
    dir_ = os.path.join(vis_path, vid) + '/'
    os.makedirs(dir_, exist_ok=True)

    with open(os.path.join(args.cluster_json_path, f"{vid}_clusters.json"), 'r') as f:
        clusters = json.load(f)

    select_img = list(clusters.keys())

    ind_idx = np.nonzero(ind_list)[0]
    frame_root = "../vce_data/frames/" + vid
    for i in ind_idx:
        idx = select_img[i]
        index = idx.split("_")[1]
        files = f'{frame_root}/{idx}'
        time = files.split("_")[-1]
        shutil.copyfile(files, dir_+"pre_"+index+"_"+time)

def test(model, loader, device, args):
    """Test function to process each video sequence."""
    model.eval()

    for input_data in tqdm(loader, total=len(loader)):
        video_seq = torch.concat(input_data['video'], dim=0).to(device, non_blocking=True)
        vid = input_data['vid'][0]

        with amp.autocast():
            with torch.no_grad():
                img_outputs, seg_outputs = model(video_seq)

            # Load features and cluster info
            senetfea = np.load(os.path.join(args.senetfea_path, f"{vid}.npy"))
            endovitfea = np.load(os.path.join(args.endovitfea_path, f"{vid}.npy"))
            
            with open(os.path.join(args.cluster_json_path, f"{vid}_clusters.json"), 'r') as f:
                clusters = json.load(f)

            representative_indices = [
                cluster['representative_index'] for cluster in clusters.values()
            ]
            
            senetfea_ = senetfea[representative_indices]
            senetfea_torch = torch.from_numpy(senetfea_).cuda()
            simi_a = torch.matmul(senetfea_torch, senetfea_torch.T).cpu().numpy()

            endovitfea_ = endovitfea[representative_indices]
            endovitfea_torch = torch.from_numpy(endovitfea_).cuda()
            simi_b = torch.matmul(endovitfea_torch, endovitfea_torch.T).cpu().numpy()

            # Select keyframes
            predict_short = Long2short(
                img_outputs, 
                simi_a, 
                simi_b, 
                seg_outputs, 
                args.model_keyframe_outputdim,
                args.model_structure_outputdim,
                args.prob_threshold,
                args.sim_threshold_a,
                args.sim_threshold_b,
                prefix=vid
            )

            IMG_ind = np.zeros((simi_a.shape[0], 1))
            IMG_ind[predict_short] = 1 

            save_frames(IMG_ind, vid, args.vis_path, args.cluster_json_path)

if __name__ == '__main__':
    args = parse_args()
    inference(args)

