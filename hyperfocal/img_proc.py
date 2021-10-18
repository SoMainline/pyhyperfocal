import cv2
import numpy as np
from typing import Tuple, Dict, TypeVar, List, Callable, Union

setting = TypeVar('setting', str, int, float, bool, List[int])


###############################################################################
# IMAGE PROCESSING STUFF
###############################################################################


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


CV_PHOTO_FILTER: Union[Callable[[np.ndarray], np.ndarray], None] = None


def process_preview(
    frame: np.ndarray,
    config: Dict[str, setting],
    canvas_res: Tuple[int, int]
) -> np.ndarray:
    global CV_PHOTO_FILTER

    if config['rotate'] != 0:
        frame = rotate_bound(frame, config['rotate'])

    # print(frame.shape[:2][::-1], canvas_res)
    if (np.array(canvas_res) <= np.array(frame.shape[:2][::-1])).any():
        frame = cv2.resize(
            frame, tuple(canvas_res)
        )

    if CV_PHOTO_FILTER is not None:
        frame = CV_PHOTO_FILTER(frame)

    return frame


def open_image_with_alpha(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3 and img.shape[2] == 4:
        img_alpha = img[:, :, 3]
        img_rgb = img[:, :, :3]
        return (img_rgb, img_alpha / 255)

    return img, np.ones(img.shape[:2], dtype=np.float32)


def process_photo(img: np.ndarray, config: Dict[str, setting]) -> np.ndarray:
    global CV_PHOTO_FILTER

    if config['rotate'] != 0:
        img = rotate_bound(img, config['rotate'])

    if CV_PHOTO_FILTER is not None:
        img = CV_PHOTO_FILTER(img)

    return img
