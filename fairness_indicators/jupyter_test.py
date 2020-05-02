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
"""Tests for Jupyter notebooks.

These run automatically in the workflow defined at
.github/workflows/jupyter.yml.
"""

import unittest
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


def get_errors(path):

  with open(path) as f:
    notebook = nbformat.read(f, as_version=4)

  proc = ExecutePreprocessor(timeout=None, kernel_name='python3')
  proc.allow_errors = True
  proc.preprocess(notebook, {'metadata': {'path': '/'}})

  errors = []
  for cell in notebook.cells:
    if 'outputs' in cell:
      for output in cell['outputs']:
        if output.output_type == 'error':
          errors.append(output)

  return errors


class TestNotebook(unittest.TestCase):

  def test_example(self):
    errors = get_errors(
        'fairness_indicators/examples/Fairness_Indicators_Example_Colab.ipynb')
    self.assertIsNone(errors, [])

  def test_facessd(self):
    errors = get_errors(
        'fairness_indicators/examples/Facessd_Fairness_Indicators_Example_Colab.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_lineage(self):
    errors = get_errors(
        'fairness_indicators/examples/Fairness_Indicators_Lineage_Case_Study.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tb(self):
    errors = get_errors(
        'fairness_indicators/examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tfco_celeba(self):
    errors = get_errors(
        'fairness_indicators/examples/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tfco_wiki(self):
    errors = get_errors(
        'fairness_indicators/examples/Fairness_Indicators_TFCO_Wiki_Case_Study.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tfhub(self):
    errors = get_errors(
        'fairness_indicators/examples/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb'
    )
    self.assertIsNone(errors, [])


if __name__ == '__main__':
  unittest.main()
