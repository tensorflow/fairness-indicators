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

import os

from setuptools import find_packages
from setuptools import setup


def select_constraint(default, nightly=None, git_master=None):
  """Select dependency constraint based on TFX_DEPENDENCY_SELECTOR env var."""
  selector = os.environ.get('TFX_DEPENDENCY_SELECTOR')
  if selector == 'UNCONSTRAINED':
    return ''
  elif selector == 'NIGHTLY' and nightly is not None:
    return nightly
  elif selector == 'GIT_MASTER' and git_master is not None:
    return git_master
  else:
    return default

REQUIRED_PACKAGES = [
    'tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3',
    'tensorflow-hub>=0.8.0,<1.0.0',
    'tensorflow-data-validation' + select_constraint(
        default='>=0.30,<0.31',
        nightly='>=0.31.0.dev',
        git_master='@git+https://github.com/tensorflow/data-validation@master'),
    'tensorflow-model-analysis' + select_constraint(
        default='>=0.30,<0.31',
        nightly='>=0.31.0.dev',
        git_master='@git+https://github.com/tensorflow/model-analysis@master'),
    'witwidget>=1.4.4,<2',
    'protobuf>=3.9.2,<4',
]

# Get version from version module.
with open('fairness_indicators/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

with open('README.md', 'r', encoding='utf-8') as fh:
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
        'fairness_indicators': ['documentation/*'],
    },
    python_requires='>=3.6,<4',
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='tensorflow model analysis fairness indicators tensorboard machine'
    ' learning',
)
