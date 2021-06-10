# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools

INSTALL_REQUIRES = [
    'numpy',
    'opencv-python',
    'torch>=1.6',
    'matplotlib',
    'pycocotools',
    'tqdm',
    'tb-nightly',
    'future',
    'Pillow',
    'thop'
]

TEST_REQUIRES = [
    'bandit',
    'flake8',
    'mypy==0.812',
    'pylint',
    'pytest-cov',
    'pytest-flake8',
    'pytest-mypy',
    'pytest-pylint',
    'pytest-xdist'
]

setuptools.setup(
    name='object_detection',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES
    },
    python_requires='>=3.7'
)
