import os
from typing import List, Tuple
import subprocess
import json
import numpy as np
import cv2


def combine_folder_paths(base_path: str, *other_paths: List[str]) -> str:
    return os.path.normpath(
        os.path.join(
            base_path,
            *other_paths
        )
    )


class Backend:
    def __init__(self, path: str):
        self.root = path
        self._list_all = combine_folder_paths(path, "list_all")
        self._get_info = combine_folder_paths(path, "get_info")
        self._setup = combine_folder_paths(path, "setup")
        self._cleanup = combine_folder_paths(path, "cleanup")

    def list_all(self) -> List[str]:
        res = subprocess.run(self._list_all, capture_output=True)

        if res.returncode:
            raise RuntimeError(f"hyperfocal.Backend failed to list all devices, reason: {res.returncode}")  # noqa PEP8 E501

        # print(res.stdout, res.returncode)

        return res.stdout.decode().split()

    def get_backend_info(self) -> str:
        res = subprocess.run(self._get_info, capture_output=True)

        if res.returncode:
            raise RuntimeError(f"hyperfocal.Backend failed to get info about the backend, reason: {res.returncode}")  # noqa PEP8 E501

        return json.loads(res.stdout.decode())

    def get_info(self, device_id: str) -> str:
        res = subprocess.run([self._get_info, device_id], capture_output=True)

        # print(res.stdout, res.returncode)

        if res.returncode:
            raise RuntimeError(f"hyperfocal.Backend failed to get info about device '{device_id}', reason: {res.returncode}")  # noqa PEP8 E501

        return res.stdout.decode()

    def setup(self, device_id: str) -> str:
        res = subprocess.run([self._setup, device_id], capture_output=True)

        # print(res.stdout, res.returncode)

        if res.returncode:
            raise RuntimeError(f"hyperfocal.Backend failed to set up device '{device_id}', reason: {res.returncode}")  # noqa PEP8 E501

        return res.stdout.decode().strip()

    def cleanup(self, device_id: str):
        res = subprocess.run([self._cleanup, device_id], capture_output=True)

        # print(res.stdout, res.returncode)

        if res.returncode:
            raise RuntimeError(f"hyperfocal.Backend failed to clean up device '{device_id}', reason: {res.returncode}")  # noqa PEP8 E501


# TODO: convert to using full fledged v4l2 instead of opencv
class VideoStream:
    def __init__(self, backend: Backend, device_id: str):
        self.device_id = device_id
        self._backend = backend

        self.v4l2_path = self._backend.setup(device_id)
        self.vd = cv2.VideoCapture(self.v4l2_path)

        ret, frame = self.vd.read()

        if not ret:
            raise RuntimeError(
                f'failed to start VideoStream "{self.v4l2_path}":"{device_id}"'
            )

        self.info = json.loads(self._backend.get_info(self.device_id))

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.vd.read()

    def stop(self):
        self.vd.release()
        self._backend.cleanup(self.device_id)

    def is_alive(self) -> bool:
        return self.vd.isOpened()
