import numpy as np


def parse_bayer_pattern(bp_str):
    num_str = bp_str.replace('R', '0').replace('G', '1').replace('B', '2')
    num_list = list(num_str)
    num_list = [int(n) for n in num_list]
    return num_list


def remosaic(rgb_im, bayer_pattern=(1, 0, 2, 1)):
    """
    Re-mosaic an RGB image,
    Default bayer pattern after mosaic:
    G r
    b G
    :param bayer_pattern: target bayer pattern
    :param rgb_im: image to mosaic. Dimensions are [h, w, c]
    :return: mosaic image, output dimension is [h, w]
    """
    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    h, w = rgb_im.shape[:-1]
    mosaic = np.zeros((h, w))
    for i in range(4):
        channel = bayer_pattern[i]
        i0, j0 = upper_left_idx[i]
        mosaic[i0::2, j0::2] = rgb_im[i0::2, j0::2, channel]
    return mosaic


def disambiguate_bp(bp):
    """
    Distinguish the 2 green channels.
    Red row G channel: 1
    Blue row G channel: 2
    """
    if bp == [0, 1, 1, 2]:
        return [0, 1, 2, 3]
    elif bp == [2, 1, 1, 0]:
        return [3, 2, 1, 0]
    elif bp == [1, 0, 2, 1]:
        return [1, 0, 3, 2]
    elif bp == [1, 2, 0, 1]:
        return [2, 3, 0, 1]
    else:
        raise Exception('Invalid bayer pattern ', bp)


def adjust_bl_wl(img, cur_bl, cur_wl, new_bl, new_wl):
    if cur_bl == new_bl and cur_wl == new_wl:
        return img

    img = img.astype(np.float32)
    img = (img - cur_bl) / (cur_wl - cur_bl)
    img = img * (new_wl - new_bl) + new_bl
    img = np.clip(img, 0, new_wl)
    return img