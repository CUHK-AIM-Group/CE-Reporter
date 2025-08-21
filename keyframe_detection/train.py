import sys
sys.path.append('../data/')
sys.path.append('../model/')
sys.path.append('../utils/')
import os
import sys
import torch
from torch.utils import data
from tensorboardX import SummaryWriter
import numpy as np
import random
from tqdm import tqdm
import time
import math
import functools
import torch.cuda.amp as amp
import torch.nn.functional as F 
from train_config import parse_args, set_path     
from data.loader_train import TrainFeatureLoader
from model.multi_task_framework import Keyframe_detect
import utils.tensorboard_utils as TB
from utils.data_utils import DataLoaderBG
from utils.train_utils import clip_gradients
from utils.utils import AverageMeter, save_checkpoint, ProgressMeter

def train(loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args):
    """
    Training loop for one epoch.
    """
    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data', ':.2f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(loader), [batch_time, data_time, losses],
        prefix='Epoch:[{}]'.format(epoch))

    model.train()
    end = time.time()
    tic = time.time()
    optimizer.zero_grad()

    # Prepare class weights for segmentation loss if provided
    weights = None
    if args.class_weights:
        weights = torch.FloatTensor(args.class_weights)
        if args.use_cuda: # Check if CUDA should be used based on args
            weights = weights.cuda(device) # Move to specific device

    loader = tqdm(loader, desc=f"Training Epoch [{epoch}]")

    for idx, input_data in enumerate(loader):
        data_time.update(time.time() - end)

        # Move data to device
        video_seq = input_data['video'].to(device, non_blocking=True)
        video_label = input_data['video_label'].to(device, non_blocking=True)
        video_seg_label = input_data['video_seg_label'].to(device, non_blocking=True)
        video_label_short = input_data['video_label_short'].to(device, non_blocking=True)
        B, T, _ = video_seq.shape

        # Forward pass with automatic mixed precision
        with amp.autocast():
            loss_dict = {}
            img_outputs, seg_outputs = model(video_seq)

            # Image prediction loss (long-term keyframe)
            loss_img = F.nll_loss(img_outputs.transpose(1, 2), video_label)
            loss_dict['loss_img'] = loss_img

            # Segmentation loss with optional class weights
            loss_seg = F.nll_loss(seg_outputs.transpose(1, 2), video_seg_label, weight=weights)
            loss_dict['loss_seg'] = loss_seg

            # Image prediction loss (short-term keyframe) ignoring index 0
            loss_img_short = F.nll_loss(img_outputs.transpose(1, 2), video_label_short, ignore_index=0)
            loss_dict['loss_img_short'] = loss_img_short

            # Combined loss using weights from args
            loss = (args.weight_keyframelong * loss_dict['loss_img'] +
                    args.weight_keyframeshort * loss_dict['loss_img_short'] +
                    args.weight_seg * loss_dict['loss_seg'])

        # Update loss meter if loss is valid
        if not torch.isinf(loss) and not torch.isnan(loss):
            losses.update(loss.item(), B)

        # Backward pass with gradient scaling
        grad_scaler.scale(loss).backward()

        # Update weights periodically based on backprop frequency
        if idx % args.backprop_freq == 0:
            grad_scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                _ = clip_gradients(model, clip_grad=args.clip_grad)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        # Log statistics to TensorBoard
        if args.iteration % 10 == 0:
            for k, v in loss_dict.items():
                args.train_plotter.add_data(f'local/{k}', v.item(), args.iteration)
            args.train_plotter.add_data('local/lr', lr_scheduler.get_last_lr()[0], args.iteration)
            args.train_plotter.add_data('device/sps', 1 / (time.time() - end), args.iteration)
            args.train_plotter.log_gpustat(step=args.iteration)
            args.train_plotter.writer.flush()

        # Update timing meters
        batch_time.update(time.time() - end)
        progress.display(idx)
        lr_scheduler.step(args.iteration) # Update learning rate
        end = time.time()
        args.iteration += 1 # Increment global iteration counter

    # Final logging for the epoch
    loader.set_description(f"Training Epoch [{epoch}] Loss: {losses.avg:.4f}")
    print(f'Epoch {epoch} finished, took {time.time() - tic:.2f} seconds')
    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    return losses.avg

@torch.no_grad()
def evaluate_downstream(loader, model, device, args, epoch):
    """
    Evaluation function for downstream tasks (validation).
    """
    model.eval() # Set model to evaluation mode
    all_metrics = {}
    losses = AverageMeter('Loss', ':.4f')

    # Prepare class weights for segmentation loss if provided
    weights = None
    if args.class_weights:
        weights = torch.FloatTensor(args.class_weights)
        if args.use_cuda: # Check if CUDA should be used based on args
            weights = weights.cuda(device) # Move to specific device

    loader = tqdm(loader, desc=f"Valid Epoch [{epoch}]")

    for idx, input_data in enumerate(loader):
        # Move data to device
        video_seq = input_data['video'].to(device, non_blocking=True)
        video_label = input_data['video_label'].to(device, non_blocking=True)
        video_seg_label = input_data['video_seg_label'].to(device, non_blocking=True)
        video_label_short = input_data['video_label_short'].to(device, non_blocking=True)
        B, T, _ = video_seq.shape

        # Forward pass (no gradient computation)
        img_outputs, seg_outputs = model(video_seq)

        # Calculate individual losses
        loss_img = F.nll_loss(img_outputs.transpose(1, 2), video_label)
        loss_seg = F.nll_loss(seg_outputs.transpose(1, 2), video_seg_label, weight=weights)
        loss_img_short = F.nll_loss(img_outputs.transpose(1, 2), video_label_short, ignore_index=0)

        # Combined loss using weights from args
        loss = (args.weight_keyframelong * loss_img +
                args.weight_keyframeshort * loss_img_short +
                args.weight_seg * loss_seg)

        # Update loss meter if loss is valid
        if not torch.isinf(loss) and not torch.isnan(loss):
            losses.update(loss.item(), B)

    # Aggregate metrics
    all_metrics.update({'val loss': losses.avg})
    return all_metrics

@torch.no_grad()
def evaluate(loader, model, device, epoch, args):
    """
    Wrapper for the evaluation function, logs metrics.
    """
    model.eval() # Ensure model is in eval mode
    metric_dict = evaluate_downstream(loader, model, device, args, epoch=epoch)

    # Log validation metrics to TensorBoard
    for k, v in metric_dict.items():
        args.val_plotter.add_data(f'metric/{k}', v, epoch)

    return metric_dict['val loss']

def setup(args):
    """
    Setup CUDA device, random seeds, and logging paths.
    """
    # DDP setting (not used in this script)
    args.distributed = int(os.environ.get('SLURM_JOB_NUM_NODES', "1")) > 1

    # CUDA setting based on args
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        device = torch.device('cuda')
        num_gpu = len(str(args.gpu).split(','))
        args.num_gpu = num_gpu
        args.batch_size = num_gpu * args.batch_size
        print(f'=> Effective BatchSize = {args.batch_size}')
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Running on CPU')

    # General setting for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.iteration = 1

    # Setup paths for logs and models
    args.log_path, args.model_path, args.exp_path = set_path(args)

    # Initialize TensorBoard writers and plotters
    writer_train = SummaryWriter(logdir=os.path.join(args.log_path, 'train'), flush_secs=60)
    args.train_plotter = TB.PlotterThread(writer_train)
    writer_val = SummaryWriter(logdir=os.path.join(args.log_path, 'val'), flush_secs=60)
    args.val_plotter = TB.PlotterThread(writer_val)

    return device

def get_dataset(args):
    """
    Create training and validation datasets and data loaders.
    """
    tokenizer = None # Not used in provided loader
    dataset_class = TrainFeatureLoader # Use the imported class

    # Create datasets
    train_dataset = dataset_class(
        video_feature_path=args.video_feature_path,
        annotation_path=args.annotation_path,
        cluster_path=args.cluster_path,
        mode='train',
        train_cases_list=args.train_case_list,
        valid_cases_list=args.valid_case_list,
        duration=args.seq_len)
    val_dataset = dataset_class(
        video_feature_path=args.video_feature_path,
        annotation_path=args.annotation_path,
        cluster_path=args.cluster_path,
        mode='val',
        train_cases_list=args.train_case_list,
        valid_cases_list=args.valid_case_list,
        duration=args.seq_len)

    # Define samplers
    train_sampler = data.RandomSampler(train_dataset)
    val_sampler = data.SequentialSampler(val_dataset)

    # Create data loaders
    train_loader = DataLoaderBG(train_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True,
        shuffle=(train_sampler is None), sampler=train_sampler,
    )
    val_loader = DataLoaderBG(val_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=False,
        shuffle=(val_sampler is None), sampler=val_sampler,
    )

    return train_dataset, val_dataset, train_loader, val_loader

def optim_policy(model, args):
    """
    Define parameter groups for the optimizer with different weight decay.
    """
    params = []
    no_decay = ['.ln_', '.bias', '.logit_scale', '.entropy_scale'] # Patterns for parameters without decay
    param_group_no_decay = []
    param_group_with_decay = []

    # Separate parameters based on decay rules
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(i in name for i in no_decay):
            param_group_no_decay.append(param)
        else:
            param_group_with_decay.append(param)

    # Add parameter groups to optimizer configuration
    params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.wd})
    return params

def main(args):
    """
    Main training loop.
    """
    # Pre-setup
    device = setup(args)
    train_dataset, val_dataset, train_loader, val_loader = get_dataset(args)

    ### Model Initialization ###
    model = Keyframe_detect(
        sim=args.sim,
        pos_enc=args.pos_enc,
        width=args.model_width,
        heads=args.model_heads,
        layers=args.model_layers,
        in_token=args.seq_len + 10, # Adjusted input token length
        text_token_len=args.model_text_token_len,
        types=args.model_types,
        fusion_layers=args.model_fusion_layers,
        keyframe_outputdim=args.model_keyframe_outputdim,
        structure_outputdim=args.model_structure_outputdim
    )
    model.to(device)
    model_without_dp = model # Reference for saving/loading state dict

    ### Optimizer Setup ###
    params = optim_policy(model, args)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    ### Resume or Pretrain from Checkpoint ###
    best_acc = 1e5 # Initialize best validation accuracy/loss
    if args.resume:
        print(f"Resuming from checkpoint {args.resume}")
        # Assuming get_model_card resolves the path/tag
        checkpoint_path = get_model_card(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args.start_epoch = checkpoint['epoch'] + 1
        args.iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']

        # Load model state dict
        try:
            model_without_dp.load_state_dict(state_dict)
        except Exception as e: # Catch specific exceptions if possible
            # Handle potential mismatch in keys
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"=" * 12}\n{chr(10).join(missing_keys)}\n{"=" * 20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"=" * 12}\n{chr(10).join(unexpected_keys)}\n{"=" * 20}')
            user_input = input('[WARNING] Non-Equal load for resuming training! Continue? [y/n]')
            if user_input.lower() == 'n':
                sys.exit()

        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.pretrain:
        print(f"Loading pretrain checkpoint {args.pretrain}")
        # Assuming get_model_card resolves the path/tag
        checkpoint_path = get_model_card(args.pretrain)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        net_dict = model_without_dp.state_dict()

        # Filter and load compatible pretrained weights
        pretrained_dict = {k: v for k, v in state_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        model_without_dp.load_state_dict(net_dict)

    ### Learning Rate Scheduler Setup ###
    args.decay_steps = args.epochs * len(train_loader)
    args.warmup_iterations = 1000

    # Cosine annealing with warmup schedule function
    def lr_schedule_fn(iteration, iter_per_epoch, args):
        if iteration < args.warmup_iterations:
            lr_multiplier = iteration / (args.warmup_iterations)
        else:
            lr_multiplier = 0.5 * (
                    1. + math.cos(math.pi * (iteration - args.warmup_iterations) /
                                  (args.epochs * iter_per_epoch - args.warmup_iterations)))
        return lr_multiplier

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_schedule_fn, iter_per_epoch=len(train_loader), args=args)
    )
    lr_scheduler.step(args.iteration) # Initialize scheduler

    ### Gradient Scaler for AMP ###
    grad_scaler = amp.GradScaler()

    print('Main training loop starts')
    # Training Loop
    for epoch in range(args.start_epoch, args.epochs):
        # Set seeds for reproducibility within epoch
        np.random.seed(epoch)
        random.seed(epoch)

        # Train for one epoch
        train_loss = train(train_loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args)

        # Evaluate on validation set
        val_loss = evaluate(val_loader, model, device, epoch, args)

        # Save checkpoint periodically or at the end
        if (epoch % args.eval_freq == 0) or (epoch == args.epochs - 1):
            is_best = val_loss < best_acc
            best_acc = min(val_loss, best_acc) # Update best accuracy/loss

            # Prepare state dict for saving
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}

            # Save checkpoint
            save_checkpoint(save_dict, is_best, args.eval_freq,
                            filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch),
                            keep_all=False)

    print(f'Training from epoch {args.start_epoch} to {args.epochs - 1} finished')
    sys.exit(0)

def get_model_card(tag):
    """
    Resolve model checkpoint paths or tags.
    Allows defining shortcuts in model_card_dict.
    """
    model_card_dict = {
        # Add your model tag shortcuts here if needed
        # e.g., "resnet50_pretrain": "/path/to/resnet50_pretrain.pth"
    }
    if tag in model_card_dict:
        print(f'Resolving model tag {tag} to path: {model_card_dict[tag]}')
    # Return resolved path or the original tag if not found in dict
    return model_card_dict.get(tag, tag)

if __name__ == '__main__':
    args = parse_args() 
    main(args)
