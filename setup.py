from setuptools import setup, find_packages
from camera_app import __version__

setup(
    name="hyperfocal",
    version=__version__,
    author="Oleg Vorobiov <okawo.198@gmail.com>",
    author_email="",
    description="python side of hobo_vr",
    long_description="long_description",
    long_description_content_type="text/markdown",
    url="https://github.com/okawo80085/hobo_vr",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy",
        "docopt",
        "opencv-python",
    ],
    extras_require={},
    entry_points={},
    python_requires=">=3.7",
)
