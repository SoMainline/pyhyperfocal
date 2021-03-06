import cv2
import numpy as np
from typing import Tuple, TypeVar, List, Callable, Union
from copy import copy

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

    target_layer = copy(CV_VISIBLE_LAYER)

    for obj in CV_VISIBLE_OBJECTS:
        if obj.layer == target_layer:
            x = obj._mouse_cb(*args)
            # print(x, obj.cb)
            if x is not None:
                break

    del target_layer


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
        layer: int = 0,
        icon_name: str = None
    ):
        global CV_VISIBLE_OBJECTS
        self.pos = np.array(pos).astype(int)
        self.img = img
        self.size = self.img.shape[:2][::-1]
        self.cb = callback
        self.layer = layer
        self.icon_name = icon_name

        self._mouse_pos = (0, 0)
        self._mouse_hold = False
        CV_VISIBLE_OBJECTS.append(self)

    def offset_by_img_size(self):
        self.pos -= (np.array(self.size) / 2).astype(int)

    def change_icon(self, img: np.ndarray):
        self.img = img
        temp_size = self.img.shape[:2][::-1]

        self.pos -= (
            np.array(temp_size) / 2 - np.array(self.size) / 2
        ).astype(int)

        self.size = temp_size

    def _mouse_cb(self, event: int, x: int, y: int, *rest) -> bool:
        # print(self.__class__.__name__, self.pos, self._mouse_pos)
        self._mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_hold = True
            self._mouse_pos = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # if self._mouse_hold:
            pass

        elif event == cv2.EVENT_LBUTTONUP:
            # this indentation crime to humanity is PEP8 compliant btw
            if (
                self.pos <= self._mouse_pos
            ).all() and (
                self.pos + self.size >= self._mouse_pos
            ).all():
                return self.cb()

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
        layer: int = 0,
        icon_name: str = None
    ):
        super().__init__(pos, img, callback, layer, icon_name)
        self.mask = np.stack((alpha_mask, ) * 3, axis=-1)
        self.mask_inv = np.stack((1 - alpha_mask, ) * 3, axis=-1)

    def change_icon(self, img: np.ndarray, alpha_mask: np.ndarray):
        super().change_icon(img)
        self.mask = np.stack((alpha_mask, ) * 3, axis=-1)
        self.mask_inv = np.stack((1 - alpha_mask, ) * 3, axis=-1)



def draw_objects(
    canvas: np.ndarray,
    frame: Union[np.ndarray, None],
    objects: List[CanvasObject]
) -> np.ndarray:
    global CV_VISIBLE_LAYER

    # draw frame
    # this indentation crime to humanity is brought to you by PEP8
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

    # draw objects
    for i in objects:
        if i.layer != CV_VISIBLE_LAYER:
            continue

        canvas[
            i.pos[1]:i.pos[1] + i.size[1],
            i.pos[0]:i.pos[0] + i.size[0],
            :
        ] = i.img

    return canvas


def draw_transparent_objects(
    canvas: np.ndarray,
    frame: Union[np.ndarray, None],
    objects: List[CanvasAlphaObject]
) -> np.ndarray:
    global CV_VISIBLE_LAYER

    # draw frame
    # this indentation crime to humanity is brought to you by PEP8
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

    # draw objects
    for i in objects:
        if i.layer != CV_VISIBLE_LAYER:
            continue

        canvas[
            i.pos[1]:i.pos[1] + i.size[1],
            i.pos[0]:i.pos[0] + i.size[0],
            :
        ] = (
            canvas[
                i.pos[1]:i.pos[1] + i.size[1],
                i.pos[0]:i.pos[0] + i.size[0],
                :
            ].astype(np.float64) * i.mask_inv
        ).astype(np.uint8)

        canvas[
            i.pos[1]:i.pos[1] + i.size[1],
            i.pos[0]:i.pos[0] + i.size[0],
            :
        ] += (i.img * i.mask).astype(np.uint8)

    return canvas
