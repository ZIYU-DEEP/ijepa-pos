# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import random
import copy
import logging
import sys
import yaml
import wandb
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.models.vision_transformer import Prober

from src.helper import (
    load_checkpoint,
    load_checkpoint_lin,
    init_model,
    init_encoder,
    init_opt)
from src.transforms import make_transforms
from torchvision import transforms
from tqdm import tqdm

# --
log_timings = True
log_freq = 20  # the iteration
checkpoint_freq = 5  # the epoch
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, port=40112, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --
    loader_name = args['data']['loader_name']

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    # -- PROBE $$$$$$$$$
    use_prober = args['probe']['use_prober']
    out_feat_keys = args['probe']['out_feat_keys']
    n_categories = args['probe']['n_categories']
    probe_interval = args['probe']['probe_interval']

    # -- POSITION $$$$$$$$$
    use_pos_predictor = args['pos']['use_pos_predictor']
    decoder_embed_dim = args['pos']['decoder_embed_dim']
    decoder_num_heads = args['pos']['decoder_num_heads']
    decoder_depth = args['pos']['decoder_depth']
    pos_drop_ratio = args['pos']['pos_drop_ratio']
    pos_lambda = args['pos']['pos_lambda']  # Control the weight for using this loss

    # -- LIN EVAL $$$$$$$$$
    r_file_encoder = args['lin']['r_file_encoder']
    resize_size = args['lin']['resize_size']
    lr_lin = args['lin']['lr_lin']
    weight_decay_lin = args['lin']['weight_decay_lin']
    momentum_lin = args['lin']['momentum_lin']
    milestones_lin = args['lin']['milestones_lin']
    gamma_lin = args['lin']['gamma_lin']
    num_epochs_lin = args['lin']['num_epochs_lin']
    log_interval_lin = args['lin']['log_interval_lin']


    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed(port=port)
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    method = 'pos' if use_pos_predictor else 'ijepa'
    folder = Path(folder) / model_name / method
    tag = f'bs{batch_size}_lr_{lr}_ep{num_epochs}_ps_{patch_size}_'
    tag += f'pr{pos_drop_ratio}_pl{pos_lambda}_'
    tag += f'sl{start_lr}_fl{final_lr}_fw{final_wd}_wp{warmup}_wd{wd}_'
    tag += f'de{decoder_embed_dim}_dh{decoder_num_heads}_dd{decoder_depth}_'
    tag += f'uc{use_color_distortion}_ug{use_gaussian_blur}_uh{use_horizontal_flip}'
    folder = folder / tag
    os.makedirs(folder, exist_ok=True)

    log_path = folder / f'lin.log'
    save_path = folder / 'lin_ep{epoch}.pth.tar'
    latest_path = folder / f'lin_latest.pth.tar'
    latest_path_encoder = folder / f'latest.pth.tar'

    # Set up the log saving
    if rank == 0:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    # Set up the wandb
    if rank == 0:
        wandb_name = f'{method}_{batch_size}_{lr}_{num_epochs}_{lr_lin}_{num_epochs_lin}'
        if use_pos_predictor: wandb_name += f'_{pos_drop_ratio}_{pos_lambda}'
        wandb.init(entity='info-ssl',
                   project=f'pos-jepa-lin-{loader_name}',
                   name=wandb_name,
                   config=args)

    logger.info(folder)

    # The path to load the lin model
    load_path = None
    if load_model:
        load_path = folder / r_file if r_file is not None else latest_path

    # The path to load the encoder model
    load_path_encoder = None 
    if r_file_encoder:
        load_path_encoder = folder / r_file_encoder
    else:
        load_path_encoder = latest_path_encoder

    # ----------------------------------------------------------- #
    # Setting the encoder
    encoder = init_encoder(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name,
        decoder_embed_dim=decoder_embed_dim,
        decoder_num_heads=decoder_num_heads,
        decoder_depth=decoder_depth,
        n_categories=n_categories)
    
    # Set the encoder in distributed mode
    encoder = DDP(encoder, static_graph=True)

    # Load the encoder
    logger.info(f'Loading weights from {load_path_encoder}')
    encoder_weights = torch.load(load_path_encoder, map_location='cpu')
    encoder.load_state_dict(encoder_weights['encoder'])

    # Set the encoder in the eval mode
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    # -----------------------------------------------------------

    # ----------------------------------------------------------- #
    # SETTING THE DATA
    # -- make data transforms
    train_transform = transforms.Compose([transforms.RandomResizedCrop(resize_size),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    
    if 'imagenet' in loader_name.lower():
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(resize_size),
                                             transforms.ToTensor()])
    else:
        test_transform = transforms.Compose([transforms.Resize(resize_size),
                                             transforms.ToTensor()])
    

    _, train_loader, train_sampler = make_imagenet1k(
            transform=train_transform,
            batch_size=batch_size,
            collator=None,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    
    _, test_loader, test_sampler = make_imagenet1k(
            transform=test_transform,
            batch_size=batch_size,
            collator=None,
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=False)
    
    ipe = len(train_loader)
    # -----------------------------------------------------------

    
    # ----------------------------------------------------------- #
    # SETTING THE LINEAR PROBING MODEL
    # -- set the in_dim
    in_dim = encoder.module.embed_dim
    for key in out_feat_keys:
        if key.startswith('concatPOOL'):
            v = int(key.replace('concatPOOL', ''))
            in_dim = encoder.module.embed_dim * v
    
        if key.startswith('lastPOOL'):
            in_dim = encoder.module.embed_dim
    
        if key.startswith('direct'):
            in_dim = encoder.module.embed_dim
    
    prober = Prober(in_dim, n_categories).to(device)
    # -----------------------------------------------------------

    # ------------------------------------------------------ #
    # Set the optimizer, scheduler and scaler
    # $$$$ TO EDIT HERE
    # THE OPTIMIZER SETTING IS EVEN WORSE THEN ONLINE PROBING
    optimizer = torch.optim.SGD(prober.parameters(),
                                lr=lr_lin,
                                weight_decay=weight_decay_lin,
                                momentum=momentum_lin,
                                nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones_lin,
        gamma=gamma_lin,
        last_epoch=-1)
    
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    # ------------------------------------------------------ 

    # ----------------------------------------------------------- #
    # SETTING THE MODEL LOADING FOR PROBER
    prober = DDP(prober)

    start_epoch = 0

    # -- load training checkpoint
    if load_model:
        prober, optimizer, scaler, start_epoch = load_checkpoint_lin(
            device=device,
            r_path=load_path,
            prober=prober,
            opt=optimizer,
            scaler=scaler)
        
        for _ in range(start_epoch*ipe):
            scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'prober': prober.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, str(save_path).format(epoch=f'{epoch + 1}'))
                logger.info(f'Model saved at {save_path}.')
    # ------------------------------------------------------ 


   # --------------------------------- RUN AN EPOCH --------------------------------- #
    def run_epoch(prober, encoder, rank, data_loader, epoch, 
                  optimizer=None, scheduler=None, scaler=None, use_bfloat16=True):

        # Set the mode
        if optimizer: 
            prober.train()
            train_sampler.set_epoch(epoch)
        else: 
            prober.eval()

        # Set the meters
        loss_meter = AverageMeter('loss')
        acc_meter = AverageMeter('acc')

        # Set the bar
        loader_bar = tqdm(data_loader)
        for x, y in loader_bar:

            # Set the bfloat
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, 
                                         enabled=use_bfloat16):
                # Set the data
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Get the features
                with torch.no_grad():
                    features = encoder(x)
                    key = out_feat_keys[0].lower()

                    if key.startswith('lastpool') or key.startswith('concatpool'):
                        features = encoder(x, 
                            out_feat_keys=out_feat_keys)[0].reshape(- 1, in_dim)
                    else:
                        features, _ = encoder(x)
                        if features.dim() == 3: 
                            features = features.mean(dim=1)

                # Just in case
                if optimizer: optimizer.zero_grad()

                # Get the logits
                logits = prober(features)

                # Get the loss
                loss = F.cross_entropy(logits, y)
                loss = AllReduce.apply(loss)

            if optimizer:
                # Zero out the gradients
                optimizer.zero_grad()
                
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()


            acc = (logits.argmax(dim=1) == y).float().mean()
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc.item(), x.size(0))

            if optimizer:
                loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                        .format(epoch, loss_meter.avg, acc_meter.avg))
                if rank == 0:
                    wandb.log({"Train Loss": loss_meter.avg, 
                               "Train Accuracy": acc_meter.avg})
            else:
                loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                        .format(epoch, loss_meter.avg, acc_meter.avg))
                if rank == 0:
                    wandb.log({"Test Loss": loss_meter.avg, 
                               "Test Accuracy": acc_meter.avg})
        
        # We use multi-step scheduler that is epoch wise
        if scheduler: scheduler.step()
        
        # Empty the cache
        torch.cuda.empty_cache()

        return loss_meter.avg, acc_meter.avg

    # Training and testing loop
    optimal_loss, optimal_acc = 1e5, 0.
    test_loss, test_acc = 0., 0.
    for epoch in range(start_epoch, num_epochs_lin):
        # ================= The training loop ================= #
        train_loss, train_acc = run_epoch(prober=prober,
                                          encoder=encoder,
                                          rank=rank,
                                          data_loader=train_loader,
                                          epoch=epoch,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          scaler=scaler,
                                          use_bfloat16=use_bfloat16)

        if rank == 0:
            wandb.log({"Lin Train Loss": train_loss, "Lin Train Acc": train_acc})

        # ================= The testing loop ================= #
        if epoch % log_interval_lin == 0 or epoch == log_interval_lin:
            test_loss, test_acc = run_epoch(prober=prober,
                                            encoder=encoder,
                                            rank=rank,
                                            data_loader=test_loader,
                                            epoch=epoch,
                                            optimizer=None,
                                            scheduler=None,
                                            scaler=None,
                                            use_bfloat16=use_bfloat16)

        # ======================= STATS ====================== #
        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc

        logger.info(f'| Epoch: {epoch}/{num_epochs_lin} '
                    f'| Train Loss: {train_loss:.4f} '
                    f'| Train Acc: {train_acc:.4f} '
                    f'| Test Loss: {test_loss:.4f} '
                    f'| Test Acc: {test_acc:.4f} '
                    f'| Best Lin Test Acc: {optimal_acc:.4f}')

        if rank == 0:
            wandb.log({'Lin Test Loss': test_loss,
                       'Lin Test Acc': test_acc,
                       'Best Lin Test Acc': optimal_acc})  

        # ======================= SAVE THINGS ====================== #          
        save_checkpoint(epoch + 1)
    
    # End the wandb
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
