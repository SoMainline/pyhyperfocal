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
# import sys
# import functools
import time
from datetime import datetime
from glob import glob
import subprocess
import pathlib
import easygui
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
    app = args['app']
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


def combine_paths(base_path: str, *other_paths: List[str]) -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(base_path),
            *other_paths
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


CV_VISIBLE_OBJECTS: List[
    TypeVar(
        'drawn_object', 'CanvasObject', 'CanvasAlphaObject'  # noqa fuck off anaconda
    )
] = []
CV_VISIBLE_LAYER: int = 0


def mouse_cb_global(*args):
    global CV_VISIBLE_OBJECTS, CV_VISIBLE_LAYER

    # print(np.array(args[1:3]) / [720, 1280])

    for obj in CV_VISIBLE_OBJECTS:
        if obj.layer == CV_VISIBLE_LAYER:
            obj._mouse_cb(*args)


# settings button callback
def set_layer(layer: int, obj: 'CanvasObject' = None):
    global CV_VISIBLE_LAYER
    CV_VISIBLE_LAYER = layer
    print(f'layer set to {layer}')
    if obj is not None:
        obj.layer = layer


class CanvasObject:
    def __init__(
        self,
        pos: Tuple[float, float],
        img: np.ndarray,
        callback: Callable,
        layer: int = 0
    ):
        global CV_VISIBLE_OBJECTS
        self.pos = np.array(pos).astype(int)
        self.img = img
        self.size = self.img.shape[:2][::-1]
        self.cb = callback
        self.layer = layer

        self._mouse_pos = (0, 0)
        self._mouse_hold = False
        CV_VISIBLE_OBJECTS.append(self)

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
        callback: Callable,
        layer: int = 0
    ):
        super().__init__(pos, img, callback, layer)
        self.mask = np.stack((alpha_mask, ) * 3, axis=-1)
        self.mask_inv = np.stack((1 - alpha_mask, ) * 3, axis=-1)


def draw_objects(
    canvas: np.ndarray,
    frame: np.ndarray,
    objects: List[CanvasObject]
) -> np.ndarray:
    global CV_VISIBLE_LAYER

    # draw frame
    # print(canvas.shape, frame.shape)
    if frame is not None:
        canvas_center = np.array(canvas.shape[:2]) // 2
        canvas_topleft = (
            canvas_center - np.array(frame.shape[:2]) / 2
        ).astype(int)

        canvas_bottomright = (
            canvas_center + np.array(frame.shape[:2]) / 2
        ).astype(int)

        # print(canvas_topleft, canvas_bottomright)
        # print(frame.shape)
        canvas[
            canvas_topleft[0]:canvas_bottomright[0],
            canvas_topleft[1]:canvas_bottomright[1],
            :
        ] = frame

    for i in objects:
        if i.layer != CV_VISIBLE_LAYER:
            continue

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
) -> np.ndarray:
    global CV_VISIBLE_LAYER

    # draw frame
    # print(canvas.shape, frame.shape)
    if frame is not None:
        canvas_center = np.array(canvas.shape[:2]) // 2
        canvas_topleft = (
            canvas_center - np.array(frame.shape[:2]) / 2
        ).astype(int)

        canvas_bottomright = (
            canvas_center + np.array(frame.shape[:2]) / 2
        ).astype(int)

        # print(canvas_topleft, canvas_bottomright)
        # print(frame.shape)
        canvas[
            canvas_topleft[0]:canvas_bottomright[0],
            canvas_topleft[1]:canvas_bottomright[1],
            :
        ] = frame

    for i in objects:
        if i.layer != CV_VISIBLE_LAYER:
            continue

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
    config: ref[Dict[str, setting]],
    gallery_button_ref: ref[CanvasObject]
) -> bool:
    save_path = save_dir

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

    image = process_photo(frame, config.get())

    cv2.imwrite(
        img_save_path,
        image
    )

    # update gallery button icon
    bt = gallery_button_ref.get()
    # this will look cursed
    # TODO: fix thumbnail transform
    bt.img = cv2.resize(
        image, bt.size, interpolation=cv2.INTER_AREA
    )
    gallery_button_ref.set(bt)

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


def _get_images(
    dir_path: str
) -> List[str]:
    image_paths = []
    img_types = ('*.jpg', '*.jpeg', '*.png')

    for i in img_types:
        image_paths += glob(f'{dir_path}/{i.lower()}')
        image_paths += glob(f'{dir_path}/{i.upper()}')

    image_paths = sorted(
        image_paths,
        key=lambda x: pathlib.Path(x).stat().st_mtime,
        reverse=True
    )

    return image_paths


def gallery(
    winname: str,
    image_save_path: str,
    canvas_res: Tuple[int, int],
    use_system_gallery: bool,
    gallery_lock_ref: ref[bool]
) -> bool:
    global _ENABLE_CALLBACKS

    _ENABLE_CALLBACKS = False

    image_paths = _get_images(image_save_path)

    if use_system_gallery:
        gallery_lock_ref.set(False)

        # this may break on other distros
        subprocess.call(('xdg-open', image_paths[0]))

        _ENABLE_CALLBACKS = True
        return True

    curr_img = cv2.imread(image_paths[0])
    cv2.imshow(winname, curr_img)

    while 1:
        cv2.imshow(winname, curr_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break

    gallery_lock_ref.set(False)
    _ENABLE_CALLBACKS = True
    return True

###############################################################################
# APP RUNTIME
###############################################################################


def main():
    args = docopt(__doc__, version=__version__)

    with open(args['<cfg_path>'], 'r') as f:
        conf = json.loads(f.read())

    app_settings, cameras = parse_args(conf)

    WINDOW_NAME = 'app'

    DEVICE_NAME = app_settings['device']  # noqa unused variable
    # SAVE_DIR = app_settings['gallery_dir']
    # SAVE_DIR = os.path.normpath(os.path.join(SAVE_DIR, 'hyperfocal'))
    DATA_DIR = combine_paths(args['<cfg_path>'], app_settings['resources_dir'])

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
        # default to the first camera if no default is given
        curr_camcfg = cameras[0]

    # print(curr_camcfg, cameras)

    vod = init_camera(curr_camcfg)

    # shared object references
    curr_camcfg_ref = ref(curr_camcfg)
    curr_camcfg_idx_ref = ref(curr_camcfg_idx)
    curr_vod_ref = ref(vod)
    camera_lock_ref = ref(False)
    gallery_lock_ref = ref(False)

    # vod = cv2.VideoCapture(cam_idx)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb_global)

    # coordinates
    # this indentation awfulness is brought to you by PEP8
    last_img_p = np.array(
        (0.25, 0.85)
    ) * app_settings['preview_resolution'] - (35, 35)
    photo_bt_p = np.array(
        (0.5, 0.85)
    ) * app_settings['preview_resolution'] - (50, 50)
    change_cam_bt_p = np.array(
        (0.75, 0.85)
    ) * app_settings['preview_resolution'] - (35, 35)

    settings_bt_p = np.array(
        (0.9, 0.07)
    ) * app_settings['preview_resolution'] - (25, 25)

    # buttons
    gallery_button = CanvasObject(
        last_img_p,
        np.ones((70, 70, 3), dtype=np.uint8) * 150,
        lambda: gallery_lock_ref.set(True)
    )

    # try to make a last photo preview if possible
    images = _get_images(app_settings['gallery_dir'])
    if len(images):
        gallery_button.img = cv2.resize(
            cv2.imread(images[0]),
            gallery_button.size,
            interpolation=cv2.INTER_AREA
        )

    gallery_button_ref = ref(gallery_button)

    take_photo_button = CanvasAlphaObject(
        photo_bt_p,
        *open_image_with_alpha(f'{DATA_DIR}/icons/photo_button.png'),
        lambda: take_photo(
            app_settings['gallery_dir'],
            curr_vod_ref,
            curr_camcfg_ref,
            gallery_button_ref
        )
    )

    settings_button = CanvasAlphaObject(
        settings_bt_p,
        *open_image_with_alpha(f'{DATA_DIR}/icons/settings_button.png'),
        None
    )

    settings_button.cb = lambda: set_layer(
        1 if CV_VISIBLE_LAYER != 1 else 0,
        settings_button  # reference to self, to stay visible
    )

    ################################################
    # layer 0 - normal app overlay
    # layer 1 - settings
    # layer 2 - resolution overlays WIP
    # layer 3 - gallery buttons WIP
    ################################################

    def gallery_toggle_setting_cb(app: Dict[str, setting]):
        if app['use_system_gallery']:
            app['use_system_gallery'] = False
        else:
            app['use_system_gallery'] = True

        print(f'use_system_gallery: {app["use_system_gallery"]}')

    def gallery_change_dir_setting_cb(app: Dict[str, setting]):
        res = easygui.diropenbox(default=True)
        if res is None:
            return

        app['gallery_dir'] = res
        print(f'changed gallery dir to "{res}"')

    def theme_change_dir_setting_cb(app: Dict[str, setting]):
        res = easygui.diropenbox(default=True)
        if res is None:
            return

        app['resources_dir'] = res
        print(f'changed theme dir to "{res}"')

    gallery_toggle_setting = CanvasAlphaObject(
        np.array(
            (0.2, 0.15)
        ) * app_settings['preview_resolution'] - (25, 25),
        *open_image_with_alpha(f'{DATA_DIR}/icons/gallery_button.png'),
        lambda: gallery_toggle_setting_cb(app_settings),
        layer=1
    )

    gallery_change_dir_setting = CanvasAlphaObject(
        np.array(
            (0.3, 0.15)
        ) * app_settings['preview_resolution'] - (25, 25),
        *open_image_with_alpha(f'{DATA_DIR}/icons/gallerydir_button.png'),
        lambda: gallery_change_dir_setting_cb(app_settings),
        layer=1
    )

    theme_change_dir_setting = CanvasAlphaObject(
        np.array(
            (0.4, 0.15)
        ) * app_settings['preview_resolution'] - (25, 25),
        *open_image_with_alpha(f'{DATA_DIR}/icons/theme_button.png'),
        lambda: theme_change_dir_setting_cb(app_settings),
        layer=1
    )

    # button lists for rendering
    buttons_opaque = [
        # layer 0
        gallery_button,
    ]

    buttons_transparent = [
        # layer 0
        take_photo_button,
        settings_button,
        # layer 1
        gallery_toggle_setting,
        gallery_change_dir_setting,
        theme_change_dir_setting,
    ]

    # don't add this function if only 1 camera is available
    if len(cameras) > 1:
        cycle_cameras_button = CanvasAlphaObject(
            change_cam_bt_p,
            *open_image_with_alpha(
                f'{DATA_DIR}/icons/change_camera_button.png'
            ),
            lambda: cycle_cameras(
                cameras,
                curr_vod_ref,
                curr_camcfg_ref,
                curr_camcfg_idx_ref,
                camera_lock_ref
            )
        )

        buttons_transparent.append(cycle_cameras_button)

    canvas_shape = (*app_settings['preview_resolution'][::-1], 3)

    # runtime loop
    while 1:

        ret, frame = curr_vod_ref.get().read()

        if not ret and not camera_lock_ref.get():
            print('video source died', curr_camcfg_idx_ref.get())
            break

        # check for a runtime switch
        # needs to be here to keep camera feed up to date
        if gallery_lock_ref.get():
            gallery(
                WINDOW_NAME,
                app_settings['gallery_dir'],
                app_settings['preview_resolution'],
                app_settings['use_system_gallery'],
                gallery_lock_ref
            )
            gallery_lock_ref.set(False)

        canvas = np.zeros(canvas_shape, dtype=np.uint8)

        frame = process_preview(
            frame, curr_camcfg_ref.get(), app_settings['preview_resolution']
        )

        image = draw_objects(canvas, frame, buttons_opaque)
        image = draw_transparent_objects(
            image, None, buttons_transparent
        )

        cv2.imshow(WINDOW_NAME, image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            # print(frame.shape)
            break

    cleanup_camera(curr_vod_ref.get(), curr_camcfg_ref.get())
    cv2.destroyAllWindows()

    # save settings
    conf['app'] = app_settings
    # conf['cameras'] =
    with open(args['<cfg_path>'], 'w') as f:
        f.write(json.dumps(conf, sort_keys=True, indent=4))


if __name__ == '__main__':
    main()
