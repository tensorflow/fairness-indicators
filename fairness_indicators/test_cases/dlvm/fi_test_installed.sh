#!/bin/bash
#
# Copyright 2021 Google LLC. All Rights Reserved.
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
#
# A script to test a Fairness Indicators JupyterLab in the current environment.
#
# Internally this script is used to test Fairness Indicators installation on DLVM/DL Container
# images.
# - https://cloud.google.com/deep-learning-vm
# - https://cloud.google.com/ai-platform/deep-learning-containers
#
# The list of the container images can be found in:
# https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container
#

notebook_test() {
  FILENAME=$1
  OUTPUT_FILENAME="results_${1}"

  if ! papermill --no-progress-bar --no-log-output "$FILENAME" "$OUTPUT_FILENAME"; then
    echo "Notebook test failed. Unable to run the test using papermill for the file: ${FILENAME}"
    exit 1
  fi
}

set -ex

PYTHON_BINARY=$(which python)

TENSORFLOW_VERSION=$(${PYTHON_BINARY} -c 'import tensorflow; print(tensorflow.__version__)')
FI_VERSION=$(${PYTHON_BINARY} -c 'import fairness_indicators; print(fairness_indicators.__version__)')

rm -rf fairness-indicators
if [[ "${FI_VERSION}" != *dev* ]]; then
  VERSION_TAG_FLAG="-b v${FI_VERSION} --single-branch"
fi


# Check FI v0.26.* with TF 2.3.*
if [[ "${TENSORFLOW_VERSION}" == 2.3.* && "${FI_VERSION}" > 0.26.* && "${FI_VERSION}" < 0.27.* ]]; then
  # The test cases is added after 0.27.0.
  git clone -b v0.27.0 --depth 1 https://github.com/tensorflow/fairness-indicators.git

# Check FI v0.27.* with TF 2.4.*
elif [[ "${TENSORFLOW_VERSION}" == 2.4.* && "${FI_VERSION}" > 0.27.* ]]; then
  if [[ "${FI_VERSION}" != *dev* ]]; then
    VERSION_TAG_FLAG="-b v${FI_VERSION} --depth 1"
  fi
  git clone ${VERSION_TAG_FLAG} https://github.com/tensorflow/fairness-indicators.git

# Let the script fail.
else
  echo "Fairness Indicators ${FI_VERSION} should not be installed on tensorflow ${TENSORFLOW_VERSION}."
  exit 1
fi

cd fairness-indicators/fairness_indicators/test_cases/dlvm/
notebook_test fairness_indicators_dlvm_test_case.ipynb
