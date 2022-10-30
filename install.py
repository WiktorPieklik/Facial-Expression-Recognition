from typing import List
from sys import platform
import pip

common = [
    'numpy',
    'pandas',
    'jupyter',
    'matplotlib'
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
        pip.main(['install', package])


if __name__ == '__main__':
    install(common)
    if platform == 'windows':
        install(windows)
    elif platform == 'darwin':
        install(darwin)
