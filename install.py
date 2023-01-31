from typing import List
from sys import platform, executable
from subprocess import check_call

common = [
    'numpy',
    'pandas',
    'jupyter',
    'matplotlib',
    'python-decouple',
    'cryptography',
    'opencv-python==4.5.5.62',  # this specific version because of lack of autocompletion in IDE
    'alive-progress',
    'simple-chalk',
    'imgaug',
    'livelossplot'
]

windows = [
    'tensorflow'
]

# remember to install tensorflow-deps!!!
darwin = [
    'tensorflow-macos',
    'tensorflow-metal'
]


def install(packages: List[str]) -> None:
    for package in packages:
        check_call([executable, '-m', 'pip', 'install', '--no-cache-dir', package])


if __name__ == '__main__':
    install(common)
    if platform == 'windows':
        install(windows)
    elif platform == 'darwin':
        install(darwin)
