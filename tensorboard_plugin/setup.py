# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup to install Fairness Indicators Tensorboard plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'protobuf >= 3.6.0',
    'setuptools >= 40.2.0',
    'tensorboard >= 2.1.0, < 3',
    'tensorflow >= 2.1.0, < 3',
    'tensorflow-model-analysis >= 0.21.2, < 1',
    # python3 specifically requires wheel 0.26
    'wheel; python_version < "3"',
    'wheel >= 0.26; python_version >= "3"',
]

with open('README.md', 'r', encoding='utf-8') as fh:
  long_description = fh.read()

setup(
    name='tensorboard_plugin_fairness_indicators',
    version='0.0.5',
    description='Fairness Indicators TensorBoard Plugin',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tensorflow/fairness-indicators',
    author='Google LLC',
    author_email='packages@tensorflow.org',
    packages=find_packages(),
    package_data={
       'tensorboard_plugin_fairness_indicators': ['static/**'],
    },
    entry_points={
        'tensorboard_plugins': [
            'fairness_indicators = tensorboard_plugin_fairness_indicators.plugin:FairnessIndicatorsPlugin',
        ],
    },
    # Disallow python 3.0 and 3.1 which lack a 'futures' module (see above).
    python_requires='>= 2.7, != 3.0.*, != 3.1.*',
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow model analysis fairness indicators tensorboard machine learning',
)
