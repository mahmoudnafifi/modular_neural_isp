import os
from tifffile import TiffFile

from .utils import parse_bayer_pattern, remosaic
from .image_synthesizer import load_noise_model, synth_noise_on_bayer

from utils.img_utils import (extract_additional_dng_metadata, extract_image_from_dng, extract_raw_metadata,
                             normalize_raw, raw_to_srgb, demosaice)
from os.path import dirname, abspath, join, basename
import numpy as np
import cv2
import time
from .constants import NOISE_MODEL_INFO

"""
Command
python -m denoising.noise_profiler.demo
"""

def raw_bayer_to_wb_rgb(bayer, metadata, out_path):
    raw_img = normalize_raw(img=bayer, black_level=metadata['black_level'], white_level=metadata['white_level'])
    img = demosaice(raw_img, cfa_pattern=metadata['pattern'])
    mat = np.eye(3)
    img = raw_to_srgb(img, illum_color=metadata['as_shot_neutral'], ccm=mat)

    img = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.imwrite(out_path, img[..., ::-1])
    return img


def raw_rgb_to_wb_rgb(raw_rgb, metadata, out_path):
    img = normalize_raw(img=raw_rgb, black_level=metadata['black_level'][0], white_level=metadata['white_level'])
    mat = np.eye(3)
    img = raw_to_srgb(img, illum_color=metadata['as_shot_neutral'], ccm=mat)

    img = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.imwrite(out_path, img[..., ::-1])
    return img


def read_denoised_dng(dng_fn):
    img = TiffFile(dng_fn)
    img.series[2].keyframe.compression = 50002  # Need this for older versions of tifffile<2024.5.10
    denoised = img.series[2].keyframe.asarray()
    return denoised


def demo_s24():
    # Params to change: file paths and choice of noise model
    iso = 1600
    clean_dng = '/Users/lucy.zhao/Data/2025_aiisp_backend/20250714_s24_dngs/img_0000_denoised.dng'
    noise_model_str = 'h-gauss-s20-v1'
    out_dir = join('/Users/lucy.zhao/Data/2025_aiisp_backend/20250710_noise_modelling', noise_model_str)
    os.makedirs(out_dir, exist_ok=True)

    noise_model_path = join(dirname(abspath(__file__)), 'noise_models', noise_model_str)
    noise_model_bl = NOISE_MODEL_INFO[noise_model_str]['black_level']
    noise_model_wl = NOISE_MODEL_INFO[noise_model_str]['white_level']
    noise_model_bp = NOISE_MODEL_INFO[noise_model_str]['bayer_pattern']
    
    noise_model = load_noise_model(path=noise_model_path)

    # Extracts metadata from DNG file.
    metadata = extract_additional_dng_metadata(clean_dng)
    metadata.update(extract_raw_metadata(clean_dng))

    # Extracts raw image from DNG file.
    raw_rgb_clean = read_denoised_dng(clean_dng)

    # Target sensor bayer pattern fixed to GRBG
    bp = [1, 0, 2, 1]  
    raw_bayer_clean = remosaic(raw_rgb_clean, bayer_pattern=bp)
    metadata['pattern'] = 'GRBG'

    # Overwrite black and white levels for denoised S24 dataset
    metadata['black_level'] = [0, 0, 0, 0]
    metadata['white_level'] = 65535.

    # Synthesize noise on clean image.
    t0 = time.time()
    raw_noisy_synth = synth_noise_on_bayer(raw_bayer_clean, 
                                           bl=metadata['black_level'][0], wl=metadata['white_level'], 
                                           bp=bp, iso=iso,
                                           noise_model=noise_model, 
                                           noise_model_bl=noise_model_bl,
                                           noise_model_wl=noise_model_wl,
                                           noise_model_bp=noise_model_bp)
    print(f'Time taken to synthesize noise on image of shape {raw_bayer_clean.shape}: {time.time() - t0:.2f} seconds.')

    # Visualize
    raw_rgb_to_wb_rgb(raw_rgb_clean, metadata, out_path=join(out_dir, basename(clean_dng[:-4]) + '_rgb_clean.png'))
    raw_bayer_to_wb_rgb(raw_bayer_clean, metadata, out_path=join(out_dir, basename(clean_dng[:-4]) + '_remosaic_clean.png'))
    raw_bayer_to_wb_rgb(raw_noisy_synth, metadata, out_path=join(out_dir, basename(clean_dng[:-4]) + f'_noisy_synth_{noise_model_str}.png'))


def demo_s24_all_noise_models():
    # Params to change: file paths
    iso = 1600
    clean_dng = '/Users/lucy.zhao/Data/2025_aiisp_backend/20250714_s24_dngs/img_0000_denoised.dng'
    out_dir = join('/Users/lucy.zhao/Data/2025_aiisp_backend/20250714_noise_modelling_results')
    os.makedirs(out_dir, exist_ok=True)

    # Extracts metadata from DNG file.
    metadata = extract_additional_dng_metadata(clean_dng)
    metadata.update(extract_raw_metadata(clean_dng))

    # Extracts raw image from DNG file.
    raw_rgb_clean = read_denoised_dng(clean_dng)

    # Target sensor bayer pattern fixed to GRBG
    bp = [1, 0, 2, 1] 
    raw_bayer_clean = remosaic(raw_rgb_clean, bayer_pattern=bp)
    metadata['pattern'] = 'GRBG'

    # Overwrite black and white levels for denoised S24 dataset
    metadata['black_level'] = [0, 0, 0, 0]
    metadata['white_level'] = 65535.  

    # Synthesize noise on clean image.
    for noise_model_str in NOISE_MODEL_INFO:
        print(noise_model_str)
        noise_model_path = join(dirname(abspath(__file__)), 'noise_models', noise_model_str)
        noise_model_bl = NOISE_MODEL_INFO[noise_model_str]['black_level']
        noise_model_wl = NOISE_MODEL_INFO[noise_model_str]['white_level']
        noise_model_bp = NOISE_MODEL_INFO[noise_model_str]['bayer_pattern']
        
        noise_model = load_noise_model(path=noise_model_path)
        t0 = time.time()
        raw_noisy_synth = synth_noise_on_bayer(raw_bayer_clean, 
                                            bl=metadata['black_level'][0], wl=metadata['white_level'], 
                                            bp=bp, iso=iso,
                                            noise_model=noise_model, 
                                            noise_model_bl=noise_model_bl,
                                            noise_model_wl=noise_model_wl,
                                            noise_model_bp=noise_model_bp)
        print(f'Time taken to synthesize noise on image of shape {raw_bayer_clean.shape}: {time.time() - t0:.2f} seconds.')
        # Visualize noisy
        raw_bayer_to_wb_rgb(raw_noisy_synth, metadata, out_path=join(out_dir, basename(clean_dng[:-4]) + f'_noisy_synth_iso{iso}_{noise_model_str}.png'))

    # Visualize clean
    raw_rgb_to_wb_rgb(raw_rgb_clean, metadata, out_path=join(out_dir, basename(clean_dng[:-4]) + '_rgb_clean.png'))
    raw_bayer_to_wb_rgb(raw_bayer_clean, metadata, out_path=join(out_dir, basename(clean_dng[:-4]) + '_remosaic_clean.png'))


def demo_bayer():
    # Params to change: file paths and choice of noise model
    iso = 3199
    clean_dng = '/Users/lucy.zhao/Data/2025_aiisp_backend/20250710_noise_modelling/024.dng'
    # noisy_dng = '/Users/lucy.zhao/Data/2025_aiisp_backend/20250710_noise_modelling/024_iso1600.dng'
    noisy_dng = '/Users/lucy.zhao/Data/2025_aiisp_backend/20250714_noise_modelling_ali_hg/HG_noise_models/Pixel9Pro_3x_ProShot/captures/20250127_171010.dng'
    noise_model_str = 'hg-Pixel9Pro_3x_ProShot-v7_Ali'
    out_dir = join('/Users/lucy.zhao/Data/2025_aiisp_backend/20250710_noise_modelling', noise_model_str)
    
    noise_model_path = join(dirname(abspath(__file__)), 'noise_models', noise_model_str)
    noise_model_bl = NOISE_MODEL_INFO[noise_model_str]['black_level']
    noise_model_wl = NOISE_MODEL_INFO[noise_model_str]['white_level']
    noise_model_bp = NOISE_MODEL_INFO[noise_model_str]['bayer_pattern']
    noise_model = load_noise_model(path=noise_model_path)

    os.makedirs(out_dir, exist_ok=True)

    # Extracts metadata from DNG file.
    meta_clean = extract_additional_dng_metadata(clean_dng)
    meta_clean.update(extract_raw_metadata(clean_dng))
    bp = parse_bayer_pattern(meta_clean['pattern'])

    # Extracts metadata from DNG file for real noisy dng.
    meta_noisy = extract_additional_dng_metadata(noisy_dng)
    meta_noisy.update(extract_raw_metadata(noisy_dng))

    # Extracts raw image from DNG file.
    raw_clean = extract_image_from_dng(clean_dng)
    raw_noisy_real = extract_image_from_dng(noisy_dng)

    # Synthesize noise on clean image.
    t0 = time.time()
    raw_noisy_synth = synth_noise_on_bayer(raw_clean, 
                                           bl=meta_clean['black_level'][0], wl=meta_clean['white_level'], 
                                           bp=bp, iso=iso,
                                           noise_model=noise_model, 
                                           noise_model_bl=noise_model_bl,
                                           noise_model_wl=noise_model_wl,
                                           noise_model_bp=noise_model_bp)
    print(f'Time taken to synthesize noise on image of shape {raw_clean.shape}: {time.time() - t0:.2f} seconds.')

    # Visualize
    raw_bayer_to_wb_rgb(raw_clean, meta_clean, out_path=join(out_dir, basename(clean_dng[:-4]) + '_clean.png'))
    raw_bayer_to_wb_rgb(raw_noisy_real, meta_noisy, out_path=join(out_dir, basename(clean_dng[:-4]) + '_noisy_real.png'))
    raw_bayer_to_wb_rgb(raw_noisy_synth, meta_clean, out_path=join(out_dir, basename(clean_dng[:-4]) + f'_noisy_synth_{noise_model_str}.png'))
    """
    Time taken to synthesize noise on image of shape (3024, 4032): 1.15 seconds.
    """


if __name__ == '__main__':
    # demo_bayer()
    # demo_s24()
    demo_s24_all_noise_models()
    
