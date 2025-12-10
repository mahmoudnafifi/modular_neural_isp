"""Credit: Ran Zhang"""

import cv2
import os
import glob
from tqdm import tqdm


ROOT = 'path/to/dataset'


mode="train" # or test
patch_size = 512
out_raw_path = f"{ROOT}/{mode}/raw_images_patches_{patch_size}"
os.makedirs(out_raw_path, exist_ok=True)
in_raw_path = f"{ROOT}/{mode}/raw_images/"
gt_path_list = [
    {"out_gt_path": f"{ROOT}/{mode}/denoised_raw_images_patches_{patch_size}",
     "in_gt_path": f"{ROOT}/{mode}/denoised_raw_images/"},
]

for gt_d in gt_path_list:
    os.makedirs(gt_d['out_gt_path'], exist_ok=True)

count = 0
for fp in tqdm(glob.glob(in_raw_path + '*.png')):
    if count % 100 == 0:
        print("processing: ", fp)
    count += 1
    
    in_img = cv2.imread(fp, -1)
    h, w, _ = in_img.shape
    img_name = fp.split('/')[-1]
    patch_positions = [(y, x) for y in range(0, h - patch_size + 1, patch_size)
                       for x in range(0, w - patch_size + 1, patch_size)]
    patch_position_right_column = [(y, w - patch_size) for y in range(0, h - patch_size + 1, patch_size)]
    patch_position_bottom_row = [(h - patch_size, x) for x in range(0, w - patch_size + 1, patch_size)]
    patch_position_bottom_right_corner = [(h - patch_size, w - patch_size)]
    patch_positions = patch_positions + patch_position_right_column + patch_position_bottom_row + patch_position_bottom_right_corner
    #print(patch_positions)
    
    for idx, gt_d in enumerate(gt_path_list):
        in_gt_path = gt_d['in_gt_path']
        out_gt_path = gt_d['out_gt_path']
        gt_img_name = os.path.join(in_gt_path, os.path.split(img_name)[-1])
        gt_img = cv2.imread(gt_img_name, -1)
    
        for y, x in patch_positions:
            patch_name = f"{os.path.split(os.path.splitext(img_name)[0])[-1]}_{y}_{x}"
            if idx == 0:

                in_patch = in_img[y:y + patch_size, x:x + patch_size, :]
                raw_patch_path = os.path.join(out_raw_path, patch_name+'.png')
                cv2.imwrite(raw_patch_path, in_patch)
            gt_patch = gt_img[y:y + patch_size, x:x + patch_size, :]
            gt_patch_path = os.path.join(out_gt_path, patch_name+'.png')
            cv2.imwrite(gt_patch_path, gt_patch)
