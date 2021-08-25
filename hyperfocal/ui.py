import cv2
import numpy as np
from typing import Tuple, TypeVar, List, Callable, Union

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
                self._mouse_pos = (x, y)

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
