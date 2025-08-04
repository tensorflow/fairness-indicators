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

import os
import sys

from setuptools import find_packages, setup

if sys.version_info >= (3, 11):
    sys.exit("Sorry, Python >= 3.11 is not supported")


def select_constraint(default, nightly=None, git_master=None):
    """Select dependency constraint based on TFX_DEPENDENCY_SELECTOR env var."""
    selector = os.environ.get("TFX_DEPENDENCY_SELECTOR")
    if selector == "UNCONSTRAINED":
        return ""
    elif selector == "NIGHTLY" and nightly is not None:
        return nightly
    elif selector == "GIT_MASTER" and git_master is not None:
        return git_master
    else:
        return default


REQUIRED_PACKAGES = [
    "protobuf>=4.21.6,<6.0.0",
    "tensorboard>=2.17.0,<2.18.0",
    "tensorflow>=2.17,<2.18",
    "tf-keras>=2.17,<2.18",
    "tensorflow-model-analysis>=0.48,<0.49",
    "werkzeug<2",
]

TEST_PACKAGES = [
    "pytest>=8.3.0,<9",
]

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from version module.
with open("tensorboard_plugin_fairness_indicators/version.py") as fp:
    globals_dict = {}
    exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict["__version__"]

setup(
    name="tensorboard_plugin_fairness_indicators",
    version=__version__,
    description="Fairness Indicators TensorBoard Plugin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorflow/fairness-indicators",
    author="Google LLC",
    author_email="packages@tensorflow.org",
    packages=find_packages(),
    package_data={
        "tensorboard_plugin_fairness_indicators": ["static/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "fairness_indicators = tensorboard_plugin_fairness_indicators.plugin:FairnessIndicatorsPlugin",
        ],
    },
    python_requires=">=3.9,<4",
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    extras_require={
        "test": TEST_PACKAGES,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="Apache 2.0",
    keywords="tensorflow model analysis fairness indicators tensorboard machine learning",
)
