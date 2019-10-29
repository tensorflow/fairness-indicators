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
"""Tests for util function to create plugin metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fairness_indicators.tensorboard_plugin import metadata
import tensorflow as tf


class MetadataTest(tf.test.TestCase):

  def testCreateSummaryMetadata(self):
    summary_metadata = metadata.CreateSummaryMetadata('description')
    self.assertEqual(metadata.PLUGIN_NAME,
                     summary_metadata.plugin_data.plugin_name)
    self.assertEqual('description', summary_metadata.summary_description)

  def testCreateSummaryMetadata_withoutDescription(self):
    summary_metadata = metadata.CreateSummaryMetadata()
    self.assertEqual(metadata.PLUGIN_NAME,
                     summary_metadata.plugin_data.plugin_name)


if __name__ == '__main__':
  tf.test.main()
