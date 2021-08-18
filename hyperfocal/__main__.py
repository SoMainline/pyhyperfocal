'''
opencv based camera app

usage:
    camera_app <cfg_path> [options]

options:
    -h, --help          show's this message
    -v, --version       show's the version of the app
'''

from . import __version__

import cv2
import numpy as np
from docopt import docopt
import json
import os
import sys
# import functools
import time
from datetime import datetime
from glob import glob
from typing import Tuple, Dict, TypeVar, List, Callable, Generic

# types
T = TypeVar('T')
app_settings = TypeVar('app_settings')
setting = TypeVar('setting', str, int, float, bool, List[int])
camera_configs = List[Dict[str, setting]]

###############################################################################
# CONFIG STUFF
###############################################################################


def parse_args(args: dict) -> Tuple[app_settings, camera_configs]:
    app = (
        args['app']['device'],
        args['app']['preview_resolution'],
        args['app']['resources_dir'],
    )

    cameras = args['cameras']

    return app, cameras


def init_camera(camera_config: Dict[str, setting]) -> cv2.VideoCapture:
    if camera_config['need_v4l2_setup']:
        raise NotImplementedError(
            f"this setup option is not suppoerted yet: 'need_v4l2_setup': {camera_config['need_v4l2_setup']}"  # noqa E501
        )

    return cv2.VideoCapture(camera_config['id'])


def cleanup_camera(vod: cv2.VideoCapture, config: Dict[str, setting]):
    if config['need_v4l2_setup']:
        raise NotImplementedError(
            f"this cleanup option is not suppoerted yet: 'need_v4l2_setup': {config['need_v4l2_setup']}"  # noqa E501
        )

    vod.release()


def validate_cameras(cameras: camera_configs) -> camera_configs:
    valid = []
    for i in cameras:
        try:
            vod = init_camera(i)

            ret, _ = vod.read()

            if not ret:
                raise RuntimeError(
                    f"failed to read frames from video source \'{i['name']}\'"
                )

            cleanup_camera(vod, i)

            valid.append(i)

        except Exception as e:
            print(f"failed to validate camera '{i['name']}': {e}")

    return valid


def combine_paths(app_config_path: str, resource_dir_path: str) -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(app_config_path),
            resource_dir_path
        )
    )

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


def process_preview(
    frame: np.ndarray,
    config: Dict[str, setting],
    canvas_res: Tuple[int, int]
) -> np.ndarray:
    if config['rotate'] != 0:
        frame = rotate_bound(frame, config['rotate'])

    # print(frame.shape[:2][::-1], canvas_res)
    if (np.array(canvas_res) <= np.array(frame.shape[:2][::-1])).all():
        frame = cv2.resize(
            frame, tuple(canvas_res), interpolation=cv2.INTER_AREA
        )

    return frame


def open_image_with_alpha(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    img_alpha = img[:, :, 3]
    img_rgb = img[:, :, :3]
    return (img_rgb, img_alpha / 255)


def process_photo(img: np.ndarray, config: Dict[str, setting]) -> np.ndarray:
    if config['rotate'] != 0:
        img = rotate_bound(img, config['rotate'])
    return img  # for now only do rotation

###############################################################################
# UI STUFF
###############################################################################


CV_MOUSE_CALLBACK_METHODS = []


def mouse_cb_global(*args):
    global CV_MOUSE_CALLBACK_METHODS
    # res = np.array([720, 1280])
    # p = np.array(args[1:3])
    # print(p / res, (res - p) / res)
    for method_cb in CV_MOUSE_CALLBACK_METHODS:
        method_cb(*args)


class CanvasObject:
    def __init__(
        self,
        pos: Tuple[float, float],
        img: np.ndarray,
        winname: str,
        callback: Callable
    ):
        global CV_MOUSE_CALLBACK_METHODS
        self.pos = np.array(pos).astype(int)
        self.img = img
        self.size = self.img.shape[:2][::-1]
        self.cb = callback

        self._mouse_pos = (0, 0)
        self._mouse_hold = False
        CV_MOUSE_CALLBACK_METHODS.append(self._mouse_cb)
        cv2.setMouseCallback(winname, mouse_cb_global)

    def _mouse_cb(self, event: int, x: int, y: int, *rest):
        # print(self.__class__.__name__, self.pos, self._mouse_pos)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_hold = True
            self._mouse_pos = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._mouse_hold:
                pass  # for now

        elif event == cv2.EVENT_LBUTTONUP:
            # this indentation crime to humanity is PEP8 compliant btw
            if (
                self.pos <= self._mouse_pos
            ).all() and (
                self.pos + self.size >= self._mouse_pos
            ).all():
                self.cb()

            # print(
            #     (
            #         self.pos <= self._mouse_pos
            #     ).all() and (
            #         self.pos + self.size >= self._mouse_pos
            #     ).all(),
            #     self.pos,
            #     self._mouse_pos
            # )
            self._mouse_hold = False


class CanvasAlphaObject(CanvasObject):
    def __init__(
        self,
        pos: Tuple[float, float],
        # expecting an rgb image
        img: np.ndarray,
        # expecting a grayscale image
        alpha_mask: np.ndarray,
        winname: str,
        callback: Callable
    ):
        super().__init__(pos, img, winname, callback)
        self.mask = np.stack((alpha_mask, ) * 3, axis=-1)
        self.mask_inv = np.stack((1 - alpha_mask, ) * 3, axis=-1)


def draw_objects(
    canvas: np.ndarray,
    frame: np.ndarray,
    objects: List[CanvasObject]
):
    # draw frame
    # print(canvas.shape, frame.shape)
    canvas[:frame.shape[0], :frame.shape[1], :] = frame

    for i in objects:
        canvas[
            i.pos[1]:i.pos[1] + i.size[0],
            i.pos[0]:i.pos[0] + i.size[1],
            :
        ] = i.img

    return canvas


def draw_transparent_objects(
    canvas: np.ndarray,
    frame: np.ndarray,
    objects: List[CanvasAlphaObject]
):
    # draw frame
    # print(canvas.shape, frame.shape)
    canvas[:frame.shape[0], :frame.shape[1], :] = frame

    for i in objects:
        canvas[
            i.pos[1]:i.pos[1] + i.size[0],
            i.pos[0]:i.pos[0] + i.size[1],
            :
        ] = (
            canvas[
                i.pos[1]:i.pos[1] + i.size[0],
                i.pos[0]:i.pos[0] + i.size[1],
                :
            ].astype(np.float64) * i.mask_inv
        ).astype(np.uint8)

        canvas[
            i.pos[1]:i.pos[1] + i.size[0],
            i.pos[0]:i.pos[0] + i.size[1],
            :
        ] += (i.img * i.mask).astype(np.uint8)

    return canvas

###############################################################################
# FUNCTIONALITY STUFF
###############################################################################


class ref(Generic[T]):
    def __init__(self, obj: T): self.obj = obj  # noqa
    def get(self):    return self.obj        # noqa
    def set(self, obj: T):      self.obj = obj  # noqa


def take_photo(
    save_dir: str,
    vod: ref[cv2.VideoCapture],
    config: ref[Dict[str, setting]]
) -> bool:
    save_path = os.path.normpath(os.path.join(save_dir, 'hyperfocal'))

    print(save_path, save_dir)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ret, frame = vod.get().read()

    if not ret:
        return False

    img_save_path = os.path.normpath(
        os.path.join(
            save_path,
            f'IMG_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_raw.png'
        )
    )

    cv2.imwrite(
        img_save_path,
        process_photo(frame, config.get())
    )

    print(f'image saved at: {img_save_path}')
    return True


def cycle_cameras(
    cameras: camera_configs,
    vod: ref[cv2.VideoCapture],
    config: ref[Dict[str, setting]],
    config_index: ref[int],
    camera_alive_lock: ref[bool]
) -> bool:
    camera_alive_lock.set(True)
    time.sleep(1 / 10)
    cleanup_camera(vod.get(), config.get())

    # print(f'last camera: {curr_camcfg_idx.get()}')
    # print(len(cameras))
    config_index.set(config_index.get() + 1)
    if len(cameras) <= config_index.get():
        config_index.set(0)

    config.set(cameras[config_index.get()])  # noqa unused variable

    print(f'current camera: {config_index.get()}')

    vod.set(init_camera(config.get()))
    camera_alive_lock.set(False)

    return True

###############################################################################
# APP RUNTIME
###############################################################################


if __name__ != '__main__':
    sys.exit(1)

args = docopt(__doc__, version=__version__)

with open(args['<cfg_path>'], 'r') as f:
    conf = json.loads(f.read())

WINDOW_NAME = 'app'

app_settings, cameras = parse_args(conf)

DEVICE_NAME = app_settings[0]
SAVE_DIR = './Camera'
DATA_DIR = combine_paths(args['<cfg_path>'], app_settings[2])

cameras = validate_cameras(cameras)

# print(cameras)

curr_camcfg = None
curr_camcfg_idx = 0

h = 0

# look for a default camera
for i in cameras:
    if i['default']:
        curr_camcfg = i
        curr_camcfg_idx = h
        break
    h += 1
else:
    curr_camcfg = cameras[0]  # default to first camera if no default is given

# print(curr_camcfg, cameras)

vod = init_camera(curr_camcfg)

cam_change_lock = False

# shared object references
curr_camcfg_ref = ref(curr_camcfg)
curr_camcfg_idx_ref = ref(curr_camcfg_idx)
curr_vod_ref = ref(vod)
camera_ret_ref = ref(cam_change_lock)

# vod = cv2.VideoCapture(cam_idx)

cv2.namedWindow(WINDOW_NAME)

# coordinates
last_img_p = np.array((0.25, 0.85)) * app_settings[1] - (35, 35)
photo_bt_p = np.array((0.5, 0.85)) * app_settings[1] - (50, 50)
change_cam_bt_p = np.array((0.75, 0.85)) * app_settings[1] - (35, 35)

settings_bt_p = np.array((0.9, 0.07)) * app_settings[1] - (25, 25)

# buttons
gallery_button = CanvasObject(
    last_img_p,
    np.ones((70, 70, 3), dtype=np.uint8) * 255,
    WINDOW_NAME,
    lambda: print('gallery')
)

take_photo_button = CanvasAlphaObject(
    photo_bt_p,
    *open_image_with_alpha(f'{DATA_DIR}/icons/photo_button.png'),
    WINDOW_NAME,
    lambda: take_photo(SAVE_DIR, curr_vod_ref, curr_camcfg_ref)
)

settings_button = CanvasAlphaObject(
    settings_bt_p,
    *open_image_with_alpha(f'{DATA_DIR}/icons/settings_button.png'),
    WINDOW_NAME,
    lambda: print('settings')
)

# button lists for rendering
buttons_opaque = [
    gallery_button,
]

buttons_transparent = [
    take_photo_button,
    settings_button,
]

# don't add this function if only 1 camera is available
if len(cameras) > 1:
    cycle_cameras_button = CanvasAlphaObject(
        change_cam_bt_p,
        *open_image_with_alpha(
            f'{DATA_DIR}/icons/change_camera_button.png'
        ),
        WINDOW_NAME,
        lambda: cycle_cameras(
            cameras,
            curr_vod_ref,
            curr_camcfg_ref,
            curr_camcfg_idx_ref,
            camera_ret_ref
        )
    )

    buttons_transparent.append(cycle_cameras_button)

canvas_shape = (*app_settings[1][::-1], 3)

# runtime loop
while 1:
    ret, frame = curr_vod_ref.get().read()

    if not ret and not camera_ret_ref.get():
        print('video source died', curr_camcfg_idx_ref.get())
        break

    canvas = np.zeros(canvas_shape, dtype=np.uint8)

    frame = process_preview(frame, curr_camcfg_ref.get(), app_settings[1])

    image = draw_objects(canvas, frame, buttons_opaque)
    image = draw_transparent_objects(
        image, np.zeros((10, 10, 3), dtype=np.uint8), buttons_transparent
    )

    cv2.imshow(WINDOW_NAME, image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        # print(frame.shape)
        break

cleanup_camera(curr_vod_ref.get(), curr_camcfg_ref.get())
cv2.destroyAllWindows()
