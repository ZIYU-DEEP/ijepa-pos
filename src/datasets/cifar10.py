import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from logging import getLogger
_GLOBAL_SEED = 0
logger = getLogger()
def make_cifar10(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    logger.info('CIFAR10 dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('CIFAR-10 data loader created')

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     collate_fn=None,  # You can add your custom collator if needed
    #     sampler=dist_sampler,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_mem,
    #     shuffle=(dist_sampler is None),  # Shuffle the data if not using DistributedSampler
    #     drop_last=True
    # )
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     collate_fn=None,  # You can add your custom collator if needed
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_mem,
    #     shuffle=False,  # No need to shuffle test data
    #     drop_last=False
    # )

    return dataset, data_loader, dist_sampler
