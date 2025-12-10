"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Zhongling Wang (z.wang2@samsung.com)
Raghav Goyal (raghav.goyal@samsung.com)
Lucy Zhao (lucy.zhao@samsung.com)

Data loader for denoising
This code is modified from the original NAFNet repository and follows the same
license as the original repo (MIT License).
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import json
import random
from os.path import abspath, dirname, join

import numpy as np
import torch
from noise_profiler.constants import NOISE_MODEL_INFO
from noise_profiler.image_synthesizer import load_noise_model, synth_noise_on_bayer
from noise_profiler.utils import parse_bayer_pattern, remosaic

from utils import img_utils as img_utils_ai_isp
from denoising.denoise_utils import get_dataset_name


class PairedImageDatasetUint16SynthNoise(torch.utils.data.Dataset):
    """
    Synthesize noise assuming bayer pattern is GRBG
    bp = [1, 0, 2, 1]
    """
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        self.is_train = True if self.split == "train" else False
        self.dataset_name = get_dataset_name(args.path_lq)

        self.paths = {
            'gt': args.path_gt,
            'lq': args.path_lq,
            'metadata': args.path_metadata
        }

        self.img_ids = sorted([e.stem for e in Path(self.paths['gt']).glob('*.png')])

        # read all metadata
        self.all_metadata = {}
        for i, id in enumerate(self.img_ids):
            if self.is_train:
                id_img_selected_original = "_".join(id.split("_")[:-2])
                metadata_path = Path(self.paths["metadata"]) / f"{id_img_selected_original}.json"
            else:
                metadata_path = Path(self.paths["metadata"]) / f"{id}.json"
            assert metadata_path.exists(), f"Metadata dir does not exist: {str(metadata_path)}"

            img_metadata_full = json.load(metadata_path.open("r"))
            img_metadata = {
                "cam_illum": np.array(img_metadata_full["cam_illum"]),
                "ccm": np.array(img_metadata_full["ccm"]),
            }
            if "additional_metadata" in img_metadata_full:
                img_metadata["original_iso"] = img_metadata_full["additional_metadata"]["iso"]
            self.all_metadata[id] = img_metadata

        print(f" > [{self.dataset_name} {self.split} synth] Num images (or crops) in {self.split} set: {len(self.img_ids)}")

        if args.use_original_iso:
            print(f" > [{self.dataset_name} {self.split} synth] Use original ISO for synthetic noise generation")
        else:
            self.iso_range = np.float32(args.iso_sampling_range)
            if self.iso_range[1] == np.inf:
                self.iso_range[1] = 3200
            self.iso_range = np.int32(self.iso_range).tolist()
            assert self.iso_range[0] <= self.iso_range[1], f"ISO range must be non-decreasing. Got {self.iso_range}"

            print(f" > [{self.dataset_name} {self.split} synth] Uniformly sample ISO from {self.iso_range} for synthetic noise generation")

        # noise synthesis
        self.noise_model_dict = self._get_noise_models()
        self.noise_model_names = list(self.noise_model_dict.keys())
        self.bp_str = 'GRBG'
        self.bp = parse_bayer_pattern(self.bp_str)


    def _get_noise_models(self):
        noise_model_dir = join(dirname(abspath(__file__)), 'noise_profiler/noise_models')
        noise_model_dict = {}
        for noise_model_str in NOISE_MODEL_INFO:
            noise_model_path = join(noise_model_dir, noise_model_str)
            noise_model = load_noise_model(path=noise_model_path)
            noise_model_dict[noise_model_str] = {
                'bl': NOISE_MODEL_INFO[noise_model_str]['black_level'],
                'wl': NOISE_MODEL_INFO[noise_model_str]['white_level'],
                'bp': NOISE_MODEL_INFO[noise_model_str]['bayer_pattern'],
                'nm': noise_model
            }
        return noise_model_dict

    def _random_crop(self, img, patch_size):
        """
        img: h, w, c
        """
        img_height, img_width = img.shape[:2]  # Get image dimensions (excluding channels if present)

        # Calculate valid range for top-left corner
        max_row = img_height - patch_size
        max_col = img_width - patch_size

        # Generate random top-left corner
        if max_row < 0 or max_col < 0:
            raise ValueError("Patch size cannot be larger than the image size.")
        start_row = np.random.randint(0, max_row)
        start_col = np.random.randint(0, max_col)

        # Make sure the starting coords are even numbers to avoid cropping the wrong bp
        if start_row % 2 == 1:
            start_row = start_row + 1
        if start_col % 2 == 1:
            start_col = start_col + 1

        # Crop the patch
        cropped_image = img[start_row:start_row + patch_size, start_col:start_col + patch_size]

        return cropped_image

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        id_img_selected = self.img_ids[index]

        gt_path = Path(self.paths["gt"]) / f"{id_img_selected}.png"
        lq_path = Path(self.paths["lq"]) / f"{id_img_selected}.png"

        img_gt = img_utils_ai_isp.imread(str(gt_path), normalize=True)  # RGB: [H, W, C]
        img_lq = remosaic(img_gt, bayer_pattern=self.bp)                # GRBG: [H, W]

        # meta data
        img_metadata = None
        if self.paths["metadata"] is not None:
            img_metadata = self.all_metadata[id_img_selected]

        if self.is_train:
            cat_img = np.concatenate([img_gt, img_lq[..., None]], axis=-1)
            cat_patch = self._random_crop(cat_img, self.args.patch_size)
            img_gt_patch = cat_patch[..., :3]
            img_lq_patch = cat_patch[..., 3:].squeeze()
        else:
            img_gt_patch = img_gt
            img_lq_patch = img_lq

        # randomly pick a noise model
        noise_model_name = random.choice(self.noise_model_names)
        noise_model = self.noise_model_dict[noise_model_name]

        if self.args.use_original_iso:
            iso = img_metadata["original_iso"]
        else:
            iso = random.randint(self.iso_range[0], self.iso_range[1])
        img_metadata["iso"] = iso

        # synthesize noise
        img_lq_patch = synth_noise_on_bayer(img_lq_patch,
                                            bl=0., wl=1.,
                                            bp=self.bp, iso=iso,
                                            noise_model=noise_model['nm'],
                                            noise_model_bl=noise_model['bl'],
                                            noise_model_wl=noise_model['wl'],
                                            noise_model_bp=noise_model['bp'])

        # demosaic
        img_lq_patch = img_utils_ai_isp.demosaice(img_lq_patch, cfa_pattern=self.bp_str)

        # h, w, c -> c, h, w
        img_lq_patch = img_lq_patch.transpose(2, 0, 1)
        img_gt_patch = img_gt_patch.transpose(2, 0, 1)

        # print(iso, 10 * np.log10(1 / np.mean((img_lq_patch - img_gt_patch) ** 2)))

        return {
            "lq": torch.from_numpy(img_lq_patch).float(),
            "gt": torch.from_numpy(img_gt_patch).float(),
            "img_metadata": img_metadata if img_metadata is not None else None,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path)
        }
