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
"""Tests for example_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import tempfile
from fairness_indicators import example_model
import six
import tensorflow.compat.v1 as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.slicer import slicer_lib as slicer

tf.compat.v1.enable_eager_execution()

TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'
SLICE = 'slice'
FEATURE_MAP = {
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    TEXT_FEATURE: tf.io.FixedLenFeature([], tf.string),
    SLICE: tf.io.VarLenFeature(tf.string),
}


class ExampleModelTest(tf.test.TestCase):

  def setUp(self):
    super(ExampleModelTest, self).setUp()
    self._base_dir = tempfile.gettempdir()

    self._model_dir = os.path.join(
        self._base_dir, 'train',
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

  def _create_example(self, comment_text, label, slice_value):
    example = tf.train.Example()
    example.features.feature[TEXT_FEATURE].bytes_list.value[:] = [
        six.ensure_binary(comment_text, 'utf8')
    ]
    example.features.feature[SLICE].bytes_list.value[:] = [
        six.ensure_binary(slice_value, 'utf8')
    ]
    example.features.feature[LABEL].float_list.value[:] = [label]
    return example

  def _create_data(self):
    examples = []
    examples.append(self._create_example('test comment', 0.0, 'slice1'))
    examples.append(self._create_example('toxic comment', 1.0, 'slice1'))
    examples.append(self._create_example('non-toxic comment', 0.0, 'slice1'))
    examples.append(self._create_example('test comment', 1.0, 'slice2'))
    examples.append(self._create_example('non-toxic comment', 0.0, 'slice2'))
    examples.append(self._create_example('test comment', 0.0, 'slice3'))
    examples.append(self._create_example('toxic comment', 1.0, 'slice3'))
    examples.append(self._create_example('toxic comment', 1.0, 'slice3'))
    examples.append(
        self._create_example('non toxic comment', 0.0, 'slice3'))
    examples.append(self._create_example('abc', 0.0, 'slice1'))
    examples.append(self._create_example('abcdef', 0.0, 'slice3'))
    examples.append(self._create_example('random', 0.0, 'slice1'))
    return examples

  def _write_tf_records(self, examples):
    data_location = os.path.join(self._base_dir, 'input_data.rio')
    with tf.io.TFRecordWriter(data_location) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
    return data_location

  def test_example_model(self):
    train_tf_file = self._write_tf_records(self._create_data())
    classifier = example_model.train_model(self._model_dir, train_tf_file,
                                           LABEL, TEXT_FEATURE, FEATURE_MAP)

    validate_tf_file = self._write_tf_records(self._create_data())
    tfma_eval_result_path = os.path.join(self._model_dir, 'tfma_eval_result')
    example_model.evaluate_model(classifier, validate_tf_file,
                                 tfma_eval_result_path, SLICE, LABEL,
                                 FEATURE_MAP)

    expected_slice_keys = [
        'Overall', 'slice:slice3', 'slice:slice1', 'slice:slice2'
    ]
    evaluation_results = tfma.load_eval_result(tfma_eval_result_path)

    self.assertLen(evaluation_results.slicing_metrics, 4)

    # Verify if false_positive_rate metrics are computed for all values of
    # slice.
    for (slice_key, metric_value) in evaluation_results.slicing_metrics:
      slice_key = slicer.stringify_slice_key(slice_key)
      self.assertIn(slice_key, expected_slice_keys)
      self.assertGreaterEqual(
          1.0, metric_value['']['']
          ['post_export_metrics/false_positive_rate@0.50']['doubleValue'])
      self.assertLessEqual(
          0.0, metric_value['']['']
          ['post_export_metrics/false_positive_rate@0.50']['doubleValue'])


if __name__ == '__main__':
  tf.test.main()
