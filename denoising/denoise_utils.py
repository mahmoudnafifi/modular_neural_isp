from typing import Union

import numpy as np
import torch


def get_dataset_name(path_lq):
    if "s24" in path_lq.lower():
        return "s24"
    elif "adobe" in path_lq.lower():
        return "adobe"
    elif "zurich" in path_lq.lower():
        return "zurich"
    elif "sidd" in path_lq.lower():
        return "sidd"
    else:
        return "unknown"


def im2single(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Converts image to 32-bit floating-point format in range [0, 1]."""
    if isinstance(img, np.ndarray):
        dtype = img.dtype
        if dtype == np.uint8:
            max_value = 255
        elif dtype == np.uint16:
            max_value = 65535
        else:
            raise ValueError(f"Unsupported numpy dtype: {dtype}")
        return img.astype(np.float32) / max_value

    elif isinstance(img, torch.Tensor):
        dtype = img.dtype
        if dtype == torch.uint8:
            max_value = 255
        elif dtype == torch.uint16:
            max_value = 65535
        else:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return img.to(torch.float32) / max_value

    else:
        raise ValueError(f"Unsupported image type: {type(img)}")
