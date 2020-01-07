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
"""Fairness Indicators Plugin Demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from tensorboard_plugin_fairness_indicators import summary_v2
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

tf.enable_eager_execution()
tf = tf2

FLAGS = flags.FLAGS

flags.DEFINE_string('eval_result_output_dir', '',
                    'Log dir containing evaluation results.')

flags.DEFINE_string('logdir', '', 'Log dir where demo logs will be written.')


def main(unused_argv):
  writer = tf.summary.create_file_writer(FLAGS.logdir)

  with writer.as_default():
    summary_v2.FairnessIndicators(FLAGS.eval_result_output_dir, step=1)
  writer.close()


if __name__ == '__main__':
  app.run(main)
