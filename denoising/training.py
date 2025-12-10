"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Raghav Goyal (raghav.goyal@samsung.com)
Zhongling Wang (z.wang2@samsung.com)

Training script for denoising module
"""


import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import argparse
import time
import json

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from denoising.nafnet_arch import NAFNet

device = torch.device('cuda')


def parse_args():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--exp_name', type=str, required=True, help='name for the experiment')
    parser.add_argument('--model_size', type=str, choices=["lite", "base", "large"], required=True, help='size of the denoising model')

    # data
    parser.add_argument('--path_gt', type=str, required=True, help='path to clean images')
    parser.add_argument('--path_lq', type=str, required=True, help='path to noisy images')
    parser.add_argument('--path_metadata', type=str, help='path to image metadata, required for synthsizing noise if use_original_iso is true')
    parser.add_argument('--patch_size', type=int, default=256, help='random cropping size for training data')

    # training params
    parser.add_argument('--total_epoch', type=int, default=100, help='num of epoch for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum learning rate for the cosline annealing lr scheduler')

    # noise synthesis
    parser.add_argument('--synth_noise', action='store_true', help='use the original iso when synthsizing noise, otherwise use uniform sampling')
    parser.add_argument('--use_original_iso', action='store_true', help='use the original iso when synthsizing noise, otherwise use uniform sampling')
    parser.add_argument('--iso_sampling_range', nargs='+', default=[0, 3200], help='range for uniformly sampling iso')
    parser.add_argument('--sampling_rates', nargs='+', default=[0.2, 0.4, 0.4], help='sampling rates for the noisy image: [clean, real noise, synthetic noise]. Sum to 1.')

    # others
    parser.add_argument('--print_freq', type=int, default=40, help='print progress every print_freq iter')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers in training dataloader')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    p_out = Path('experiments') / args.exp_name
    p_out.mkdir(exist_ok=True, parents=True)

    # data
    if args.synth_noise:
        from dataset_hybrid import HybridPairedImageDataset as PairedImageDatasetUint16
    else:
        from dataset import PairedImageDatasetUint16        # equivalent to setting synth_noise=True and sampling_rates=[0, 1, 0]
    dataset_train = PairedImageDatasetUint16(args, split="train")
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # model
    model_config_path = Path("configs") / f"{args.model_size.lower()}.json"
    with open(str(model_config_path)) as f:
        model_config = json.load(f)

    model = NAFNet(
        width=model_config['width'],
        middle_block_num=model_config['middle_block_num'],
        encoder_block_nums=model_config['encoder_block_nums'],
        decoder_block_nums=model_config['decoder_block_nums'],
    )

    num_total_param = sum(p.numel() for p in model.parameters())
    print(f" > Total params: {num_total_param / 1e6:.3f} M")

    model = model.to(device)

    loss_fn = nn.L1Loss()

    total_iter = len(dataset_train) * args.total_epoch // args.batch_size
    iter_per_epoch = len(dataloader_train)  # number of iterations per epoch

    if args.lr_min > args.lr:
        args.lr_min = args.lr / 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=[0.9, 0.9])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=args.lr_min)

    tb_log_dir = Path('experiments') / args.exp_name / "tensorboard"
    tb_log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # train
    steps = 0
    latest_ep = -1
    for epoch in range(latest_ep + 1, latest_ep + 1 + args.total_epoch):
        print(f" > Training | Epoch {epoch}/{latest_ep + args.total_epoch} | Num steps per epoch: {len(dataloader_train)}")
        model.train()

        epoch_start_time = time.time()
        for current_iter, batch_train in enumerate(dataloader_train):
            optimizer.zero_grad()
            pred = model(batch_train['lq'].to(device))

            loss = loss_fn(pred, batch_train['gt'].to(device))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            scheduler.step()

            steps += 1

            # log
            if steps % args.print_freq == 0:
                msg = f"epoch: {epoch}/{args.total_epoch} || iter: {current_iter}/{iter_per_epoch} || lr: {scheduler.get_last_lr()[0]:.6f} || loss: {loss.item():.4f}"
                print(msg)
                writer.add_scalar("train/loss", loss.item(), steps)

        print(f"{len('Training |') * ' '} Epoch train time: {time.time() - epoch_start_time:.2f}s")

        # saving
        p_out.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), p_out / f"ep_{epoch}.pth")


if __name__ == '__main__':
    main()
