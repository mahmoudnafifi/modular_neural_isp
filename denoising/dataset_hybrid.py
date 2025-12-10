"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Zhongling Wang (z.wang2@samsung.com)
Raghav Goyal (raghav.goyal@samsung.com)

Hybrid dataset for denoising that mixes:
- (clean, clean) pairs
- (real noisy, clean) pairs
- (synthetic noisy, clean) pairs
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from denoising.dataset import PairedImageDatasetUint16
from denoising.dataset_synth import PairedImageDatasetUint16SynthNoise


class HybridPairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        self.is_train = True if self.split == "train" else False

        # Normalize sampling rates to sum to 1
        total = sum(self.args.sampling_rates)
        self.sampling_rates = np.array(self.args.sampling_rates) / total
        self.cum_probs = np.cumsum(self.sampling_rates)

        # initialize datasets with non-zero sampling rates
        if self.sampling_rates[1] > 0:     # sampling_rates: [clean, real noise, synth noise]
            self.real_dataset = PairedImageDatasetUint16(args, split)
        else:
            self.real_dataset = None

        if self.sampling_rates[2] > 0:
            self.synth_dataset = PairedImageDatasetUint16SynthNoise(args, split)
        else:
            self.synth_dataset = None

    def __len__(self):
        # All datasets should have same length since they use same gt images
        if self.real_dataset is not None:
            return len(self.real_dataset)
        return len(self.synth_dataset)

    def __getitem__(self, index):
        # Randomly select pair type based on sampling rates
        rand_val = random.random()
        if rand_val < self.cum_probs[0]:  # clean-clean pair
            # Get clean image from either available dataset
            if self.real_dataset is not None:
                clean_data = self.real_dataset[index]
            else:
                clean_data = self.synth_dataset[index]
            return {
                'lq': clean_data['gt'],
                'gt': clean_data['gt'],
                'img_metadata': clean_data['img_metadata'],
                'lq_path': clean_data['gt_path'],
                'gt_path': clean_data['gt_path']
            }
        elif rand_val < self.cum_probs[1]:  # real noisy
            if self.real_dataset is None:
                raise ValueError("Requested real noisy pair but real dataset is not initialized")
            return self.real_dataset[index]
        else:  # synthetic noisy
            if self.synth_dataset is None:
                raise ValueError("Requested synthetic noisy pair but synthetic dataset is not initialized")
            return self.synth_dataset[index]
