# Lint as: python2, python3
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
"""Tests for Fairness Indicators summary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

# Standard imports
from tensorboard_plugin_fairness_indicators import metadata
from tensorboard_plugin_fairness_indicators import summary_v2
import six
import tensorflow.compat.v1 as tf
from tensorboard.compat import tf2

try:
  tf2.__version__  # Force lazy import to resolve
except ImportError:
  tf2 = None

try:
  tf.enable_eager_execution()
except AttributeError:
  # TF 2.0 doesn't have this symbol because eager is the default.
  pass


class SummaryV2Test(tf.test.TestCase):

  def _write_summary(self, eval_result_output_dir):
    writer = tf2.summary.create_file_writer(self.get_temp_dir())
    with writer.as_default():
      summary_v2.FairnessIndicators(eval_result_output_dir, step=1)
    writer.close()

  def _get_event(self):
    event_files = sorted(glob.glob(os.path.join(self.get_temp_dir(), '*')))
    self.assertEqual(len(event_files), 1)
    events = list(tf.train.summary_iterator(event_files[0]))
    # Expect a boilerplate event for the file_version, then the summary one.
    self.assertEqual(len(events), 2)
    return events[1]

  def testSummary(self):
    self._write_summary('output_dir')
    event = self._get_event()

    self.assertEqual(1, event.step)

    summary_value = event.summary.value[0]
    self.assertEqual(metadata.PLUGIN_NAME, summary_value.tag)
    self.assertEqual(
        'output_dir',
        six.ensure_text(summary_value.tensor.string_val[0], 'utf-8'))
    self.assertEqual(metadata.PLUGIN_NAME,
                     summary_value.metadata.plugin_data.plugin_name)


if __name__ == '__main__':
  tf.test.main()
