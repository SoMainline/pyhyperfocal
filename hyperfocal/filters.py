import cv2
import numpy as np
# from typing import Tuple, Dict, TypeVar, List, Callable, Union

# setting = TypeVar('setting', str, int, float, bool, List[int])

###############################################################################
# IMAGE FILTERS
###############################################################################


def times1000(img: np.ndarray) -> np.ndarray:
    return (img * 1000).astype(np.uint8)


def times2000(img: np.ndarray) -> np.ndarray:
    return (img * 10000).astype(np.uint8)


def power2(img: np.ndarray) -> np.ndarray:
    return (img ** 2).astype(np.uint8)


def grayscale(img: np.ndarray) -> np.ndarray:
    return np.stack((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ) * 3, axis=-1)

###############################################################################
# VIDEO FILTERS (WIP... since there is no video processing yet...)
###############################################################################
