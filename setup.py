from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

setup(
    name='RoboCupEnv',
    py_modules=['spinup'],
    version="0.0.1",
    install_requires=[
        'numpy'
    ],
    description="LearnYouASoccer",
    author="Oswin, Kyle, Will",
)
