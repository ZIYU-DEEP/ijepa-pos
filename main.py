# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint

import torch
import yaml

from src.utils.distributed import init_distributed
from src.newtrain import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/cifar10.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    # torch.distributed.init_process_group(
    #     world_size=world_size,
    #     rank=rank)
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = 1 #len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()
#
# import argparse
# import pprint
# import logging
# import torch
# import yaml
# from src.newtrain import main as app_main
#
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--fname', type=str,
#     help='name of config file to load',
#     default='/home/christina/PycharmProjects/ijepa/configs/cifar10.yaml'
# )
# parser.add_argument(
#     '--device', type=str, default='cuda',
#     help='which device to use (cuda or cpu)'
# )
#
#
# def process_main(fname, device):
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(device.split(':')[-1])
#
#     logging.basicConfig()
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#
#     logger.info(f'called-params {fname}')
#
#     # Load script params
#     params = None
#     with open(fname, 'r') as y_file:
#         params = yaml.load(y_file, Loader=yaml.FullLoader)
#         logger.info('loaded params...')
#         pp = pprint.PrettyPrinter(indent=4)
#         pp.pprint(params)
#
#     # Run the main training function
#     # torch.distributed.init_process_group(
#     #             backend='nccl',
#     #             world_size=1,
#     #             rank=0)
#     app_main(args=params)
#
#
# if __name__ == '__main__':
#     args = parser.parse_args()
#
#     # Set the device
#     device = args.device
#
#     # Run the training on the single GPU
#     process_main(args.fname, device)
