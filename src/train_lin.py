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
    load_model_encoder = args['lin']['load_path_encoder']
    r_file_encoder = args['lin']['r_file_encoder']


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
    log_file = folder / f'lin_r{rank}.csv'
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
        wandb_name = f'{method}_{batch_size}_{lr}_{num_epochs}'
        if use_pos_predictor: wandb_name += f'_{pos_drop_ratio}_{pos_lambda}'
        wandb.init(entity='info-ssl',
                   project=f'pos-jepa-{loader_name}',
                   name=wandb_name,
                   config=args)

    logger.info(folder)

    # The path to load the lin model
    load_path = None
    if load_model:
        load_path = folder / r_file if r_file is not None else latest_path

    # The path to load the encoder model
    load_path_encoder = None 
    if load_model_encoder:
        if r_file_encoder:
            load_path_encoder = folder / r_file_encoder
        else:
            load_path_encoder = latest_path_encoder

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

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
    # Setting the data
    # -- make data transforms


    # $$$ TO EDIT HERE
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
    

    # -- init data-loaders/samplers
    # $$$ TO EDIT HERE
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
            drop_last=True)
    
    ipe = len(train_loader)

    # -- init optimizer and scheduler
    # $$$ TO EDIT HERE, just follow vissl setting
    optimizer = torch.optim.SGD(prober.parameters(),
                                lr=0.01,
                                weight_decay=1e-4,
                                momentum=0.9,
                                nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15, 30, 45],
        gamma=0.1,
        last_epoch=-1)
    
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    

    # SET THE LINEAR PROBING MODEL
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
    # $$$$ TO Set the optimizer and scheduler here
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
            'prober': encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, str(save_path).format(epoch=f'{epoch + 1}'))
                logger.info(f'Model saved at {save_path}.')

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        # New
        probe_loss_meter = AverageMeter()
        probe_acc_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(train_loader):

            # --------------------------------------------------------------- #
            # Loading image for this iteration
            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                imgs_targets = udata[1].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, imgs_targets, masks_1, masks_2)

            imgs, imgs_targets, masks_enc, masks_pred = load_imgs()
            # ---------------------------------------------------------------

            # --------------------------------------------------------------- #
            # Training for this iteration
            def train_step():
                _new_lr = scheduler.step()
                # --

                # ----------------------------------------------------------- #
                # Original I-JEPA objective
                def forward_target():
                    with torch.no_grad():
                        # h = target_encoder(imgs)
                        h, _ = target_encoder(imgs)  # [probe] _ are logits detached for probing
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z, logits = encoder(imgs, masks_enc)  # [probe] logits are detached
                    z = predictor(z, masks_enc, masks_pred)
                    return z, logits

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    return loss
                # ----------------------------------------------------------- 

                # ----------------------------------------------------------- #
                # Enhancement with position prediction
                def pos_forward_context():
                    z, logits, pos_logits, pos_bool, pos_targets = encoder(imgs, 
                                                                   masks_enc, 
                                                                   pos_drop_ratio,
                                                                   use_pos_predictor=True)
                    # Notice that this z has pos_embed partially dropped
                    z = predictor(z, masks_enc, masks_pred)

                    return z, logits, pos_logits, pos_bool, pos_targets

                def pos_loss_fn(z, h, pos_logits, pos_bool, pos_targets):
                    # ---------------------------------------------------- #
                    # The IJEPA Lokss
                    ijepa_loss = F.smooth_l1_loss(z, h)
                    # ----------------------------------------------------

                    # ---------------------------------------------------- #
                    # The Position Prediction Loss
                    # Get the no. of patches -> essentially no. of classes
                    num_patches = pos_logits.size(-1)
                    
                    # # Only calculate for logits whose positions are dropped
                    # pos_logits = pos_logits[pos_bool.unsqueeze(-1).expand(
                    #     -1, -1, num_patches)].reshape(
                    #         batch_size, -1, num_patches)  
                    
                    # # Get the loss
                    # # # We can do position smoothing + attentive reconstruction
                    # # # But let's just use the basic cross entropy for now
                    # # # Maybe we should take its mean instead of permute?
                    # # # Or multiply the mask instead of doing slicing?
                    # pos_loss = F.cross_entropy(pos_logits.permute(0, 2, 1), pos_targets) 

                    pos_bool = pos_bool.unsqueeze(-1).expand_as(pos_logits)
                    pos_logits = pos_logits[pos_bool].view(-1, num_patches)
                    pos_targets = pos_targets.view(-1)
                    pos_loss = F.cross_entropy(pos_logits, pos_targets)
                    # ----------------------------------------------------           

                    return ijepa_loss, pos_loss
                # ----------------------------------------------------------- 

                # --------------------------------------------------------------- #
                # Probing for this iteration
                def probe_loss_acc(logits, imgs_targets):
                    # [probe] logits are detached
                    probe_loss = F.cross_entropy(logits, imgs_targets)

                    # [issue] double check this part
                    probe_acc = (logits.argmax(dim=1) == imgs_targets).float().mean()
                    return probe_loss, probe_acc
                # ---------------------------------------------------------------

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    # Get the target
                    h = forward_target()

                    # ------------------------------------------
                    # Original i-jepa objective
                    if not use_pos_predictor:
                        # Get the forward pass for predictor
                        z, logits = forward_context()

                        # Get the ijepa loss
                        loss = loss_fn(z, h)
                        
                        # Just to hold the name
                        ijepa_loss, pos_loss = loss, 0  

                    # I-jepa + position prediction
                    else:
                        # Get the forward pass for both predictor and pos predictor
                        z, logits, pos_logits, pos_bool, pos_targets = pos_forward_context()

                        # Get the ijepa loss and the pos loss
                        ijepa_loss, pos_loss = pos_loss_fn(z, h, pos_logits, pos_bool, pos_targets)
                        loss = ijepa_loss + pos_lambda * pos_loss

                    # Get the detached probe loss
                    probe_loss, probe_acc = probe_loss_acc(logits, imgs_targets)
                    
                    # Sum up the loss
                    loss_all = loss + probe_loss  # [probe] logits are detached from encoder
                    loss_all = AllReduce.apply(loss_all)


                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss_all).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_all.backward()
                    optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), float(ijepa_loss), float(pos_loss), float(probe_loss), float(probe_acc),
                        _new_lr, _new_wd, grad_stats)

            # Apply the train step function
            (loss, ijepa_loss, pos_loss, probe_loss, probe_acc, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)

            # Update the meters
            loss_meter.update(loss)
            time_meter.update(etime)
            probe_loss_meter.update(probe_loss)
            probe_acc_meter.update(probe_acc)
            # ---------------------------------------------------------------


            # --------------------------------------------------------------- #
            # Logging for this iteration
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):

                    # Log to local
                    logger.info('[%d, %5d] [loss: %.3f] '
                                '[probe loss: %.3f] '
                                '[probe acc: %.4f] '
                                # '[masks: %.1f %.1f] '
                                '[wd: %.2e] [lr: %.2e] '
                                # '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   probe_loss_meter.avg,
                                   probe_acc_meter.avg,
                                #    maskA_meter.avg,
                                #    maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                #    torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))
                    # Log to wandb
                    if rank == 0:
                        wandb.log({'loss': loss_meter.avg,
                                   'probe loss': probe_loss_meter.avg,
                                   'probe acc': probe_acc_meter.avg,
                                  })

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'
            # ---------------------------------------------------------------

        # -- Save Checkpoint after every epoch
        logger.info(f'avg. loss {loss_meter.avg:.3f}')
        logger.info(f'avg. probe loss {probe_loss_meter.avg:.3f}')
        logger.info(f'avg. probe acc {probe_acc_meter.avg:.4f}')
        save_checkpoint(epoch + 1)

        # For wandb 
        if rank == 0:
            wandb.log({'avg. loss': loss_meter.avg,
                       'avg. probe loss': probe_loss_meter.avg,
                       'avg. probe acc': probe_acc_meter.avg,
                        })
        
    
    # End the wandb
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
