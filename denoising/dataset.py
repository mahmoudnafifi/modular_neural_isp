"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Raghav Goyal (raghav.goyal@samsung.com)
Zhongling Wang (z.wang2@samsung.com)

Data loader for denoising
This code is modified from the original NAFNet repository and follows the same
license as the original repo (MIT License).
"""


import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import json

import numpy as np
import torch
from torchvision.transforms import v2

from denoising.denoise_utils import get_dataset_name, im2single
from utils import img_utils as img_utils_ai_isp


class PairedImageDatasetUint16(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        self.is_train = True if self.split == "train" else False
        self.is_patch = True if args.path_gt.find("patch") != -1 else False
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
            if self.is_train and self.is_patch:
                id_img_selected_original = "_".join(id.split("_")[:-2])
                metadata_path = Path(self.paths["metadata"]) / f"{id_img_selected_original}.json"
            else:
                metadata_path = Path(self.paths["metadata"]) / f"{id}.json"
            assert metadata_path.exists(), f"Metadata dir does not exist: {str(metadata_path)}"

            img_metadata_full = json.load(metadata_path.open("r"))

            if self.paths["metadata"].find("sidd") == -1:
                # s24, zurich
                img_metadata = {
                    "cam_illum": np.array(img_metadata_full["cam_illum"]),
                    "ccm": np.array(img_metadata_full["ccm"]),
                }
                if "additional_metadata" in img_metadata_full:
                    img_metadata["iso"] = img_metadata_full["additional_metadata"]["iso"]
                    img_metadata["original_iso"] = img_metadata["iso"]      # hybrid mode compatibility
            else:
                # sidd
                img_metadata = {"iso": img_metadata_full["iso"],
                                "cam_illum": np.array(img_metadata_full["AsShotNeutral"])[0],
                                "color_matrix_1": np.array(img_metadata_full["ColorMatrix1"]).reshape([3, 3]),
                                "color_matrix_2": np.array(img_metadata_full["ColorMatrix2"]).reshape([3, 3]),
                                "calib_illum_1": np.array(img_metadata_full["CalibrationIlluminant1"])[0][0],
                                "calib_illum_2": np.array(img_metadata_full["CalibrationIlluminant2"])[0][0]
                                }

            self.all_metadata[id] = img_metadata

        print(f" > [{self.dataset_name} {self.split}] Num images (or crops) in {self.split} set: {len(self.img_ids)}")

        if self.is_train:
            self.transform = v2.Compose([v2.RandomCrop(args.patch_size)])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        id_img_selected = self.img_ids[index]

        gt_path = Path(self.paths['gt']) / f"{id_img_selected}.png"
        lq_path = Path(self.paths['lq']) / f"{id_img_selected}.png"

        img_gt = torch.tensor(img_utils_ai_isp.imread(str(gt_path), normalize=False).copy()).to(torch.uint16).permute(2, 0, 1)
        img_lq = torch.tensor(img_utils_ai_isp.imread(str(lq_path), normalize=False).copy()).to(torch.uint16).permute(2, 0, 1)

        # torch.uint16 -> torch.float32
        img_gt = im2single(img_gt)
        img_lq = im2single(img_lq)

        # meta data
        img_metadata = None
        if self.paths["metadata"] is not None:
            img_metadata = self.all_metadata[id_img_selected]

        if self.is_train:
            # random crop
            gt_lq = torch.cat((img_gt, img_lq), dim=0)
            gt_lq = self.transform(gt_lq)
            img_gt_patch_normed, img_lq_patch_normed = gt_lq[:3], gt_lq[3:]
        else:
            img_gt_patch_normed = img_gt
            img_lq_patch_normed = img_lq

        return {
            'lq': img_lq_patch_normed,
            'gt': img_gt_patch_normed,
            'img_metadata': img_metadata if img_metadata is not None else None,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
        }
