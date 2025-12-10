"""
Author(s):
Abdelrahman Abdelhamed
Lucy Zhao
"""
import os
import numpy as np

from denoising.noise_profiler.utils import adjust_bl_wl, disambiguate_bp


def adjust_channel_ordering(dst_params, src_bp, tgt_bp):
    """
    Adjust the channel ordering of the b1 b2 params to match with the bayer pattern
    of a target sensor.

    :param dst_params: noise model b1 b2 params of each of the 4 color channels, shape (4, 2) 
    :param src_bp: source bayer pattern, the pattern of the data used for noise model calibration
    :param tgt_bp: target bayer pattern, the pattern of the sensor on which the noise model will be applied
    """
    src_bp = disambiguate_bp(src_bp)
    tgt_bp = disambiguate_bp(tgt_bp)
    new_dst_params = np.empty_like(dst_params)
    for tgt_ch_idx in range(4):
        tgt_color_ch = tgt_bp[tgt_ch_idx]
        src_ch_idx = src_bp.index(tgt_color_ch)
        new_dst_params[tgt_ch_idx] = dst_params[src_ch_idx]
    return new_dst_params


def synthesize_noisy_image_v2(src_image, model, min_val=None, max_val=None, dst_iso=None,
                              per_channel=True, fix_exp=None, spat_var=False,
                              iso2b1_interp_splines=None,
                              iso2b2_interp_splines=None,
                              noise_model_bp=None,
                              tgt_sensor_bp=None,
                              black_diff=0):
    """
    Synthesize a noisy image from `src_image` using a heteroscedastic Gaussian noise model `model`.
    Add noise to the unnormalized bayer, directly read off of the DNG.
    Input values are mostly in the range of [black_level, white_level].

    Assumption: 
    The channel ordering of noise model params -- dst_params and iso2b1_interp_splines, 
    iso2b2_interp_splines must align with the bayer pattern. The channels of the noise model params 
    go in the order of top left, top right, bottom left, bottom right of each 2x2 bayer cell in the bayer image. 

    :param src_image: Clean/Semi-clean bayer. Unnormalized. Not demosaiced.
    :param metadata: Image metadata.
    :param model: Noise model.
    :param src_iso: ISO of `src_image`.
    :param dst_iso: ISO of noisy image to be synthesized.
    :param min_val: Minimum image intensity.
    :param max_val: Maximum image intensity.
    :param dst_image: If not None, match the brightness with provided destination image `dst_image`.
    :param debug: Whether to perform some debugging steps.
    :param per_channel: Whether to apply model per channel.
    :param fix_exp: Whether to fix image exposure before and after noise synthesis.
    :param match_bright_gt: Optionally, match the brightness with provided source image `src_image`.
    :param spat_var: Simulate spatial variations in noise.
    :param cross_corr: Simulate spatial cross correlations of noise.
    :param cov_mat_fn: Filename of covariance matrix used to simulate spatial correlations.
    :param iso2b1_interp_splines: Interpolation/extrapolation splines for shot noise (beta1)
    :param iso2b2_interp_splines: Interpolation/extrapolation splines for read noise (beta2)
    :param noise_model_bp: Bayer pattern of the images used for noise model calibration
    :param tgt_sensor_bp: Bayer pattern of src_image on which the noise will be synthesized.
    :param black_diff: Difference in black level from the original image and the synthesized noisy image.
        (e.g., original_black - synthetic_black).
    :return: Synthesized noisy image, and optionally, a another copy of the image with brightness matching `dst_image`.
    """
    # make a copy
    image = src_image.copy().astype(np.float32)

    if fix_exp is not None:
        # update image exposure to match target ISO before synthesizing noise, then re-scale it back later
        image_fix_exp = np.round(image.astype(np.float) * fix_exp).astype(np.uint32)
    else:
        image_fix_exp = None

    # if target ISO is not specified, select a random value
    if dst_iso is None:
        dst_iso = np.random.randint(50, 3201)

    if iso2b1_interp_splines is None or iso2b2_interp_splines is None:
        iso2b1_interp_splines = model['iso2b1_interp_splines']
        iso2b2_interp_splines = model['iso2b2_interp_splines']

    # get noise params (shot, read), per channel; LZ: dst_params, shape (4, 2), 4 channels, b1, b2
    if dst_iso in model:
        dst_params = model[dst_iso]
    else:
        dst_params = np.zeros((4, 2))
        for c in range(4):
            dst_params[c, 0] = iso2b1_interp_splines[c](dst_iso)
            dst_params[c, 1] = iso2b2_interp_splines[c](dst_iso)

    if noise_model_bp is not None and tgt_sensor_bp is not None:
        dst_params = adjust_channel_ordering(dst_params, 
                                             src_bp=noise_model_bp, 
                                             tgt_bp=tgt_sensor_bp)
    # compute noise variance, std. dev.
    if per_channel:
        dst_var = np.zeros(shape=image.shape)
        bp_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for ch in range(4):
            i0 = bp_idx[ch][0]
            j0 = bp_idx[ch][1]
            if fix_exp is not None:
                dst_var[i0::2, j0::2] = image_fix_exp[i0::2, j0::2] * dst_params[ch, 0] + dst_params[ch, 1]
            else:
                # Fix: account for difference in black level between original image and synthetic noisy image
                dst_var[i0::2, j0::2] = (image[i0::2, j0::2] + black_diff) * dst_params[ch, 0] + dst_params[ch, 1]
    else:
        dst_var = image * dst_params[0] + dst_params[1]

    # simulate variance of noise variance
    if spat_var:
        dst_var[dst_var < 0] = 0
        dst_var += np.random.normal(loc=0, scale=1, size=image.shape) * np.sqrt(dst_var)

    dst_var[dst_var < 0] = 0

    # std. dev.
    dst_std = np.sqrt(dst_var)

    # Normal Gaussian noise
    noise = np.random.normal(loc=0, scale=1, size=image.shape)

    # scale by heteroscedastic standard deviation
    noise *= dst_std

    if fix_exp is not None:
        noise /= fix_exp

    # add noise
    noisy_image = (image + noise).astype(image.dtype)

    # clip
    noisy_image = np.clip(noisy_image, min_val, max_val)

    return noisy_image


def load_noise_model(path):
    """
    Load noise model.
    :param path: Path to noise model: either a directory to a non-mixture model or a file path to a mixture model.
    :return: Corresponding noise parameter data.
    """
    assert os.path.isdir(path)
    noise_model_path = os.path.join(path, 'model_params.npy')
    iso2b1_interp_splines_fn = os.path.join(path, 'iso2b1_interp_splines.npy')
    iso2b2_interp_splines_fn = os.path.join(path, 'iso2b2_interp_splines.npy')
    noise_model, iso2b1_interp_splines, iso2b2_interp_splines = load_model(noise_model_path, 
                                                                           iso2b1_interp_splines_fn, 
                                                                           iso2b2_interp_splines_fn)
    noise_model_obj = {
        'noise_model': noise_model,
        'iso2b1_interp_splines': iso2b1_interp_splines,
        'iso2b2_interp_splines': iso2b2_interp_splines,
    }
    return noise_model_obj


def load_model(noise_model_path, iso2b1_interp_splines_fn, iso2b2_interp_splines_fn):
    """
    Load noise model files.
    :param noise_model_path: Path to model directory.
    :param iso2b1_interp_splines_fn: Shot noise interpolation spline file name.
    :param iso2b2_interp_splines_fn: Read noise interpolation spline file name.
    :return: Noise discrete parameters, shot noise interpolation spline, read noise interpolation spline.
    """
    model_arr = np.load(noise_model_path)
    model_ = dict()
    for im_idx in range(model_arr.shape[0] // 4):
        model_[model_arr[im_idx * 4, 0]] = model_arr[im_idx * 4:(im_idx + 1) * 4, 2:4]
    # load interpolation/extrapolation splines for noise parameters
    iso2b1_interp_splines = np.load(iso2b1_interp_splines_fn, allow_pickle=True)
    iso2b2_interp_splines = np.load(iso2b2_interp_splines_fn, allow_pickle=True)
    return model_, iso2b1_interp_splines, iso2b2_interp_splines


def get_noise_params_interp(iso2b1_interp_splines, iso2b2_interp_splines, dst_iso):
    """
    Get noise parameters (shot, read) per channel, for a given ISO.
    :param iso2b1_interp_splines: Shot noise interpolation spline.
    :param iso2b2_interp_splines: Read noise interpolation spline.
    :param dst_iso: Target ISO level.
    :return: Noise parameters.
    """
    noise_params = np.zeros((4, 2))
    for c in range(4):
        noise_params[c, 0] = iso2b1_interp_splines[c](dst_iso)
        noise_params[c, 1] = iso2b2_interp_splines[c](dst_iso)
    return noise_params


def synth_noise_on_bayer(bayer, bl, wl, bp,
                         iso,
                         noise_model,
                         noise_model_bl, noise_model_wl, noise_model_bp):
    """
    :param bayer: Clean/Semi-clean bayer. Unnormalized. Not demosaiced.
    :param bl: black level of the input bayer 
    :param wl: white level of the input bayer 
    :param bp: bayer pattern of the input bayer as a list of numbers, example: [0, 1, 1, 2]
    :param iso: ISO of noisy image to be synthesized.
    :param noise_model_obj: noise model b1 b2 parameters and iso interpolation splines 
    :param noise_model_bl: black level of the images used for noise model calibration
    :param noise_model_wl: white level of the images used for noise model calibration
    :param noise_model_bp: bayer pattern of the images used for noise model calibration 
    as a list of numbers
    """
    bayer_clean = adjust_bl_wl(bayer,
                             cur_bl=bl, cur_wl=wl,
                             new_bl=noise_model_bl, new_wl=noise_model_wl)
    bayer_noisy = synthesize_noisy_image_v2(bayer_clean, model=noise_model['noise_model'],
                                          dst_iso=iso, min_val=0, max_val=noise_model_wl,
                                          iso2b1_interp_splines=noise_model['iso2b1_interp_splines'],
                                          iso2b2_interp_splines=noise_model['iso2b2_interp_splines'],
                                          noise_model_bp=noise_model_bp,
                                          tgt_sensor_bp=bp)
    bayer_noisy = adjust_bl_wl(bayer_noisy,
                             cur_bl=noise_model_bl, cur_wl=noise_model_wl,
                             new_bl=bl, new_wl=wl)
    return bayer_noisy

