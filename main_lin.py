# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train_lin import main as app_main

import socket
import errno


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def find_free_port():
    """
    Tries to find a free port on the localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.
    

def process_main(rank, fname, world_size, devices, port):
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

    world_size, rank = init_distributed(port=port,
                                        rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params, port=port)


if __name__ == '__main__':
    args = parser.parse_args()
    port = find_free_port()
    print(f'Find free port {port}.')

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, port)
        ).start()
