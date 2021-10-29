from setuptools import setup, find_packages
from hyperfocal import __version__

setup(
    name="hyperfocal",
    version=__version__,
    author="Oleg Vorobiov <oleg.vorobiov@somainline.org>",
    author_email="",
    description="A camera app made for Linux",
    long_description="long_description",
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy",
        "docopt",
        "opencv-python",
        "easygui",
    ],
    extras_require={},
    entry_points={"console_scripts": ["hyperfocal=hyperfocal.__main__:main"]},
    python_requires=">=3.7",
)
