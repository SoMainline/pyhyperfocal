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
from datetime import datetime
from glob import glob
import subprocess
import pathlib
import easygui
import textwrap
from typing import Tuple, Dict, TypeVar, List, Generic, Union, Callable

from . import ui
from . import img_proc
from . import filters
from . import cam

# types
T = TypeVar('T')
app_settings = TypeVar('app_settings')
setting = TypeVar('setting', str, int, float, bool, List[int])
camera_configs = List[Dict[str, setting]]


def combine_paths(base_path: str, *other_paths: List[str]) -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(base_path),
            *other_paths
        )
    )

###############################################################################
# FUNCTIONALITY STUFF
###############################################################################


class ref(Generic[T]):
    def __init__(self, obj: T): self.obj = obj  # noqa
    def get(self):    return self.obj        # noqa
    def set(self, obj: T):      self.obj = obj  # noqa


def take_photo(
    save_dir: str,
    vod: ref[cam.VideoStream],
    gallery_button: ui.CanvasObject
) -> bool:

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ret, frame = vod.get().read()

    if not ret:
        return False

    img_save_path = os.path.normpath(
        os.path.join(
            save_dir,
            f'IMG_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_raw.png'
        )
    )

    image = img_proc.process_photo(frame, vod.get().info)

    cv2.imwrite(
        img_save_path,
        image
    )

    gallery_button.img = cv2.resize(
        image, gallery_button.size, interpolation=cv2.INTER_AREA
    )

    print(f'image saved at: {img_save_path}')
    return True


def cycle_cameras(
    cameras: List[str],
    vod: ref[cam.VideoStream],
    backend: cam.Backend,
    camera_alive_lock: ref[bool]
) -> bool:

    camera_alive_lock.set(True)

    try:
        vod.get().stop()

        old_id = vod.get().device_id

        new_index = cameras.index(old_id) + 1
        if len(cameras) == new_index:
            new_index = 0

        new_id = cameras[new_index]
        vod.set(cam.VideoStream(backend, new_id))

    except Exception as e:
        print(f'Could not cycle cameras, reason: {e}')

    camera_alive_lock.set(False)

    return True


def _get_image_paths(
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
    app_settings: Dict[str, setting],
    ROOT_DIR: str
) -> bool:

    DATA_DIR = combine_paths(ROOT_DIR, app_settings['resources_dir'])
    gallery_dir = combine_paths(ROOT_DIR, app_settings['gallery_dir'])

    image_paths = _get_image_paths(gallery_dir)

    if app_settings['use_system_gallery']:
        # this may break on other distros
        subprocess.call(('xdg-open', image_paths[0]))

        ui.set_layer(0)
        return True

    curr_img = ref(cv2.imread(image_paths[0]))
    curr_img_idx = ref(0)
    # cv2.imshow(winname, curr_img)

    killer_ref = ref(False)

    def next_img(image_paths: List[str], idx: ref[int], img: ref[np.ndarray]):
        if idx.get() + 1 >= len(image_paths):
            return

        idx.set(idx.get() + 1)

        temp = cv2.imread(image_paths[idx.get()])

        old_img = img.get()
        del old_img

        img.set(temp)

        print('next image', temp.shape)

    def prev_img(image_paths: List[str], idx: ref[int], img: ref[np.ndarray]):
        if idx.get() - 1 < 0:
            return

        idx.set(idx.get() - 1)

        temp = cv2.imread(image_paths[idx.get()])

        old_img = img.get()
        del old_img

        img.set(temp)

        print('next image', temp.shape)

    canvas_shape = (*app_settings['preview_resolution'][::-1], 3)

    buttons_transparent = [
        # layer 3  - gallery
        ui.CanvasAlphaObject(
            np.array(
                (0.1, 0.1)
            ) * app_settings['preview_resolution'] - (25, 25),
            *img_proc.open_image_with_alpha(
                f'{DATA_DIR}/icons/back_button.png'
            ),
            lambda: killer_ref.set(True),
            layer=3
        ),
        ui.CanvasAlphaObject(
            np.array((0, 0)),
            np.zeros((canvas_shape[0], canvas_shape[1] // 2, 3)),
            np.zeros((canvas_shape[0], canvas_shape[1] // 2)),
            lambda: prev_img(image_paths, curr_img_idx, curr_img),
            layer=3
        ),
        ui.CanvasAlphaObject(
            np.array((app_settings['preview_resolution'][0] // 2, 0)),
            np.zeros((canvas_shape[0], canvas_shape[1] // 2, 3)),
            np.zeros((canvas_shape[0], canvas_shape[1] // 2)),
            lambda: next_img(image_paths, curr_img_idx, curr_img),
            layer=3
        ),
    ]

    while 1:
        canvas = np.zeros(canvas_shape, dtype=np.uint8)

        frame = img_proc.process_preview(
            curr_img.get(),
            {'rotate': 0},  # dummy camera config
            app_settings['preview_resolution']
        )

        image = ui.draw_transparent_objects(canvas, frame, buttons_transparent)

        cv2.imshow(winname, image)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q') or killer_ref.get():
            break

    ui.set_layer(0)
    return True

###############################################################################
# APP RUNTIME
###############################################################################


def main():

    ###########################################################################
    # arg parse
    ###########################################################################
    args = docopt(__doc__, version=__version__)

    with open(args['<cfg_path>'], 'r') as f:
        app_settings = json.loads(f.read())

    # app_settings, cameras = parse_args(conf)

    WINDOW_NAME = 'app'

    DEVICE_NAME = app_settings['device']  # noqa unused variable
    # SAVE_DIR = app_settings['gallery_dir']
    # SAVE_DIR = os.path.normpath(os.path.join(SAVE_DIR, 'hyperfocal'))
    DATA_DIR = combine_paths(args['<cfg_path>'], app_settings['resources_dir'])

    hyperfocal_backend = cam.Backend(app_settings['backend_dir'])

    camera_ids = hyperfocal_backend.list_all()

    vod = ref(
        cam.VideoStream(
            hyperfocal_backend,
            camera_ids[0]
        )
    )

    camera_lock_ref = ref(False)
    canvas_shape = (*app_settings['preview_resolution'][::-1], 3)

    # vod = cv2.VideoCapture(cam_idx)

    # setting up app window
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, ui.mouse_cb_global)

    ################################################
    # layer 0 - normal app overlay
    # layer 1 - settings
    # layer 2 - resolution overlays WIP
    # layer 3 - gallery buttons
    ################################################

    # main buttons

    gallery_button = ui.CanvasAlphaObject(
        np.array(
            (0.25, 0.85)
        ) * app_settings['preview_resolution'] - (35, 35),
        np.ones((70, 70, 3), dtype=np.uint8) * 150,
        cv2.circle(
            np.zeros((70, 70), dtype=np.uint8),
            (35, 35),
            35,
            255,
            -1
        ) / 255,
        lambda: ui.set_layer(3)
    )

    # try to make a last photo preview if possible
    images_paths = _get_image_paths(app_settings['gallery_dir'])
    if len(images_paths):
        gallery_button.img = cv2.resize(
            cv2.imread(images_paths[0]),
            gallery_button.size,
            interpolation=cv2.INTER_AREA
        )

    take_photo_button = ui.CanvasAlphaObject(
        np.array(
            (0.5, 0.85)
        ) * app_settings['preview_resolution'] - (50, 50),
        *img_proc.open_image_with_alpha(f'{DATA_DIR}/icons/photo_button.png'),
        lambda: take_photo(
            app_settings['gallery_dir'],
            curr_vod_ref,
            curr_camcfg_ref,
            gallery_button
        )
    )

    settings_button = ui.CanvasAlphaObject(
        np.array(
            (0.9, 0.07)
        ) * app_settings['preview_resolution'] - (25, 25),
        *img_proc.open_image_with_alpha(
            f'{DATA_DIR}/icons/settings_button.png'
        ),
        None
    )

    settings_button.cb = lambda: ui.set_layer(
        1 if ui.CV_VISIBLE_LAYER != 1 else 0,
        settings_button  # reference to self, to stay visible
    )

    ################################################
    # layer 0 - normal app overlay
    # layer 1 - settings
    # layer 2 - resolution overlays WIP
    # layer 3 - gallery buttons
    ################################################

    # settings tab

    def gallery_toggle_setting_cb(
        app: Dict[str, setting],
        button_ref: ui.CanvasAlphaObject
    ) -> bool:
        if app['use_system_gallery']:
            app['use_system_gallery'] = False
            button_ref.mask *= 0.5
        else:
            app['use_system_gallery'] = True
            button_ref.mask /= 0.5

        print(f'use_system_gallery: {app["use_system_gallery"]}')
        return True

    def gallery_change_dir_setting_cb(
        app: Dict[str, setting],
        gallery_button: ui.CanvasAlphaObject
    ) -> bool:

        res = easygui.diropenbox(default=True, msg="choose gallery dir")
        if res is None:
            return False

        app['gallery_dir'] = res
        print(f'changed gallery dir to "{res}"')

        # if there are any images, load the first one for preview
        img_paths = _get_image_paths(app['gallery_dir'])
        if img_paths:
            img = img_proc.open_image_with_alpha(
                img_paths[0]
            )
            # only update the image, not the mask
            gallery_button.img = cv2.resize(
                img[0], gallery_button.size, interpolation=cv2.INTER_AREA
            )

        return True

    def theme_change_dir_setting_cb(app: Dict[str, setting]) -> bool:
        res = easygui.diropenbox(default=True, msg="choose theme dir")
        if res is None:
            return False

        app['resources_dir'] = res
        print(f'changed theme dir to "{res}"')
        easygui.msgbox(
            'You need to restart to apply the changes.', 'Restart required'
        )

        return True

    OVERLAY_IMG: Tuple[np.ndarray, np.ndarray] = None

    def toggle_grid(
        app: Dict[str, setting],
        button: ui.CanvasAlphaObject
    ) -> bool:
        nonlocal OVERLAY_IMG

        if OVERLAY_IMG:
            button.mask *= 0.5
            app['use_grid'] = False
            OVERLAY_IMG = None

        else:
            button.mask /= 0.5
            app['use_grid'] = True
            img_pair = img_proc.open_image_with_alpha(
                f'{DATA_DIR}/imgs/grid.png'
            )
            img_mask = np.stack((img_pair[1], ) * 3, axis=-1)
            OVERLAY_IMG = (img_pair[0], img_mask)

        return True

    if app_settings['use_grid']:
        img_pair = img_proc.open_image_with_alpha(
            f'{DATA_DIR}/imgs/grid.png'
        )
        img_mask = np.stack((img_pair[1], ) * 3, axis=-1)
        OVERLAY_IMG = (img_pair[0], img_mask)

    # prep to save the gallery toggle icon state
    button_icon = img_proc.open_image_with_alpha(
        f'{DATA_DIR}/icons/gallery_button.png'
    )

    # make it dimmer if its off
    if not app_settings['use_system_gallery']:
        alpha_mask = button_icon[1]
        alpha_mask *= 0.5
        button_icon = (button_icon[0], alpha_mask)

    gallery_toggle_setting = ui.CanvasAlphaObject(
        np.array(
            (0.2, 0.15)
        ) * app_settings['preview_resolution'] - (25, 25),
        *button_icon,
        None,  # assigned below
        layer=1
    )

    gallery_toggle_setting.cb = lambda: gallery_toggle_setting_cb(
        app_settings,
        gallery_toggle_setting  # self ref to update the icon on click
    )

    gallery_change_dir_setting = ui.CanvasAlphaObject(
        np.array(
            (0.3, 0.15)
        ) * app_settings['preview_resolution'] - (25, 25),
        *img_proc.open_image_with_alpha(
            f'{DATA_DIR}/icons/gallerydir_button.png'
        ),
        lambda: gallery_change_dir_setting_cb(app_settings, gallery_button),
        layer=1
    )

    theme_change_dir_setting = ui.CanvasAlphaObject(
        np.array(
            (0.4, 0.15)
        ) * app_settings['preview_resolution'] - (25, 25),
        *img_proc.open_image_with_alpha(f'{DATA_DIR}/icons/theme_button.png'),
        lambda: theme_change_dir_setting_cb(app_settings),
        layer=1
    )

    grid_button_icon = img_proc.open_image_with_alpha(
        f'{DATA_DIR}/icons/grid_button.png'
    )

    if not app_settings['use_grid']:
        alpha_mask = grid_button_icon[1]
        alpha_mask *= 0.5
        grid_button_icon = (grid_button_icon[0], alpha_mask)

    toggle_grid_setting = ui.CanvasAlphaObject(
        np.array(
            (0.5, 0.15)
        ) * app_settings['preview_resolution'] - (25, 25),
        *grid_button_icon,
        None,
        layer=1
    )
    toggle_grid_setting.cb = lambda: toggle_grid(
        app_settings, toggle_grid_setting
    )

    # has to be created here
    settings_anywhere_button = ui.CanvasAlphaObject(
        (0, 0),
        np.zeros(canvas_shape, dtype=np.uint8),
        np.zeros(canvas_shape[:2], dtype=np.float32),
        lambda: ui.set_layer(0, settings_button),
        layer=1
    )

    ################################################
    # layer 0 - normal app overlay
    # layer 1 - settings
    # layer 2 - resolution overlays WIP
    # layer 3 - gallery buttons
    ################################################

    def set_filter(filt: Union[Callable[[np.ndarray], np.ndarray], None]):
        img_proc.CV_PHOTO_FILTER = filt
        print(f'filter set to: {filt}')

    # filters
    text_pos = (1, 15)

    clear_filter = ui.CanvasAlphaObject(
        np.array(
            (0.02, 0.75)
        ) * app_settings['preview_resolution'] - (0, 15),
        cv2.putText(
            np.zeros((30, 80, 3), dtype=np.uint8),
            "clear",
            text_pos,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255)
        ),
        np.ones((30, 80)) * 0.4,
        lambda: set_filter(None)
    )

    t1000_filter = ui.CanvasAlphaObject(
        np.array(
            (0.14, 0.75)
        ) * app_settings['preview_resolution'] - (0, 15),
        cv2.putText(
            np.zeros((30, 80, 3), dtype=np.uint8),
            "times1000",
            text_pos,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255)
        ),
        np.ones((30, 80)) * 0.4,
        lambda: set_filter(filters.times1000)
    )

    t2000_filter = ui.CanvasAlphaObject(
        np.array(
            (0.26, 0.75)
        ) * app_settings['preview_resolution'] - (0, 15),
        cv2.putText(
            np.zeros((30, 80, 3), dtype=np.uint8),
            "times2000",
            text_pos,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255)
        ),
        np.ones((30, 80)) * 0.4,
        lambda: set_filter(filters.times2000)
    )

    power2_filter = ui.CanvasAlphaObject(
        np.array(
            (0.38, 0.75)
        ) * app_settings['preview_resolution'] - (0, 15),
        cv2.putText(
            np.zeros((30, 80, 3), dtype=np.uint8),
            "power2",
            text_pos,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255)
        ),
        np.ones((30, 80)) * 0.4,
        lambda: set_filter(filters.power2)
    )

    gray_filter = ui.CanvasAlphaObject(
        np.array(
            (0.50, 0.75)
        ) * app_settings['preview_resolution'] - (0, 15),
        cv2.putText(
            np.zeros((30, 80, 3), dtype=np.uint8),
            "gray",
            text_pos,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255)
        ),
        np.ones((30, 80)) * 0.4,
        lambda: set_filter(filters.grayscale)
    )

    ################################################
    # button lists for rendering
    ################################################

    buttons_opaque = [
        # layer 0 - main overlay
    ]

    buttons_transparent = [
        # layer 0 - main overlay
        gallery_button,
        take_photo_button,
        settings_button,
        clear_filter,
        t1000_filter,
        t2000_filter,
        power2_filter,
        gray_filter,
        # layer 1 - settings
        settings_anywhere_button,
        gallery_toggle_setting,
        gallery_change_dir_setting,
        theme_change_dir_setting,
        toggle_grid_setting,
    ]

    # don't add this function if only 1 camera is available
    if len(cameras) > 1:
        cycle_cameras_button = ui.CanvasAlphaObject(
            np.array(
                (0.75, 0.85)
            ) * app_settings['preview_resolution'] - (35, 35),
            *img_proc.open_image_with_alpha(
                f'{DATA_DIR}/icons/change_camera_button.png'
            ),
            lambda: cycle_cameras(
                camera_ids,
                vod,
                hyperfocal_backend,
                camera_lock_ref
            )
        )

        buttons_transparent.append(cycle_cameras_button)

    #############################################
    # main runtime loop
    #############################################

    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) > 0:

        ret, frame = vod.get().read()

        if not ret and not camera_lock_ref.get():
            print(f'video source "{vod.get()}" died')
            break

        # check for a runtime switch
        # needs to be here to keep camera feed up to date
        if ui.CV_VISIBLE_LAYER == 3:
            gallery(
                WINDOW_NAME,
                app_settings,
                args['<cfg_path>']
            )

        canvas = np.zeros(canvas_shape, dtype=np.uint8)

        frame = img_proc.process_preview(
            frame,
            vod.get().info,
            app_settings['preview_resolution']
        )

        image = ui.draw_objects(canvas, frame, buttons_opaque)
        image = ui.draw_transparent_objects(
            image, None, buttons_transparent
        )

        # temp draw overlay logic
        if OVERLAY_IMG is not None:
            canvas_center = np.array(image.shape[:2]) // 2
            canvas_topleft = (
                canvas_center - np.array(OVERLAY_IMG[0].shape[:2]) / 2
            ).astype(int)

            canvas_bottomright = (
                canvas_center + np.array(OVERLAY_IMG[0].shape[:2]) / 2
            ).astype(int)

            image[
                canvas_topleft[0]:canvas_bottomright[0],
                canvas_topleft[1]:canvas_bottomright[1],
                :
            ] = (
                image[
                    canvas_topleft[0]:canvas_bottomright[0],
                    canvas_topleft[1]:canvas_bottomright[1],
                    :
                ].astype(np.float64) * (1 - OVERLAY_IMG[1])
            ).astype(np.uint8)

            image[
                canvas_topleft[0]:canvas_bottomright[0],
                canvas_topleft[1]:canvas_bottomright[1],
                :
            ] += (OVERLAY_IMG[0] * OVERLAY_IMG[1]).astype(np.uint8)

        cv2.imshow(WINDOW_NAME, image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            # print(frame.shape)
            break

    hyperfocal_backend.cleanup(vod.get().device_id)
    cv2.destroyAllWindows()

    # save settings
    with open(args['<cfg_path>'], 'w') as f:
        f.write(json.dumps(app_settings, sort_keys=True, indent=4))


if __name__ == '__main__':
    main()
