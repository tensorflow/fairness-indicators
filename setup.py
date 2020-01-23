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
"""Setup to install Fairness Indicators."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'setuptools >= 40.2.0',
    'tensorflow >= 1.15, < 3',
    'tensorflow-model-analysis >= 0.21.0, < 1',
    'tensorflow-data-validation >= 0.15.0, < 1',
    'witwidget >= 1.4.4, < 2',
    # python3 specifically requires wheel 0.26
    'wheel; python_version < "3"',
    'wheel >= 0.26; python_version >= "3"',
]

# Get version from version module.
with open('fairness_indicators/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

with open('README.md', 'r') as fh:
  long_description = fh.read()

setup(
    name='fairness_indicators',
    version=__version__,
    description='Fairness Indicators',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tensorflow/fairness-indicators',
    author='Google LLC',
    author_email='packages@tensorflow.org',
    packages=find_packages(exclude=['tensorboard_plugin']),
    package_data={
        'fairness_indicators': [
            'examples/*.ipynb', 'examples/*.md', 'documentation/*'
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
    keywords='tensorflow model analysis fairness indicators tensorboard machine'
    ' learning',
)
