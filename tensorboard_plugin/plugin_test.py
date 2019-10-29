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
"""Tests the Tensorboard Fairness Indicators plugin."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import google3
from fairness_indicators.tensorboard_plugin import plugin
from fairness_indicators.tensorboard_plugin import summary_v2
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from werkzeug import test as werkzeug_test
from werkzeug import wrappers
from google3.third_party.tensorboard.backend import application
from google3.third_party.tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer
from google3.third_party.tensorboard.plugins import base_plugin

try:
  # python version >= 3.3
  from unittest import mock  # pylint: disable=g-import-not-at-top,g-importing-member
except ImportError:
  import mock  # pylint: disable=g-import-not-at-top,unused-import

tf.compat.v1.enable_eager_execution()
tf = tf.compat.v2


class PluginTest(tf.test.TestCase):
  """Tests for Fairness Indicators plugin server."""

  def setUp(self):
    super(PluginTest, self).setUp()
    # Log dir to save temp events into.
    self._log_dir = self.get_temp_dir()
    self._eval_result_output_dir = os.path.join(self.get_temp_dir(),
                                                "eval_result")
    if not os.path.isdir(self._eval_result_output_dir):
      os.mkdir(self._eval_result_output_dir)

    writer = tf.summary.create_file_writer(self._log_dir)

    with writer.as_default():
      summary_v2.FairnessIndicators(self._eval_result_output_dir, step=1)
    writer.close()

    # Start a server that will receive requests.
    self._multiplexer = event_multiplexer.EventMultiplexer({
        ".": self._log_dir,
    })
    self._context = base_plugin.TBContext(
        logdir=self._log_dir, multiplexer=self._multiplexer)
    self._plugin = plugin.FairnessIndicatorsPlugin(self._context)
    self._multiplexer.Reload()
    wsgi_app = application.TensorBoardWSGI([self._plugin])
    self._server = werkzeug_test.Client(wsgi_app, wrappers.BaseResponse)
    self._routes = self._plugin.get_plugin_apps()

  def tearDown(self):
    super(PluginTest, self).tearDown()
    shutil.rmtree(self._log_dir, ignore_errors=True)

  def _exportEvalSavedModel(self, classifier):
    temp_eval_export_dir = os.path.join(self.get_temp_dir(), "eval_export_dir")
    _, eval_export_dir = classifier(None, temp_eval_export_dir)
    return eval_export_dir

  def _writeTFExamplesToTFRecords(self, examples):
    data_location = os.path.join(self.get_temp_dir(), "input_data.rio")
    with tf.io.TFRecordWriter(data_location) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
    return data_location

  def _makeExample(self, age, language, label):
    example = tf.train.Example()
    example.features.feature["age"].float_list.value[:] = [age]
    example.features.feature["language"].bytes_list.value[:] = [
        language.encode("utf8")
    ]
    example.features.feature["label"].float_list.value[:] = [label]
    return example

  def testRoutes(self):
    self.assertIsInstance(self._routes["/get_evaluation_result"],
                          collections.Callable)
    self.assertIsInstance(self._routes["/index.js"], collections.Callable)
    self.assertIsInstance(self._routes["/vulcanized_tfma.js"],
                          collections.Callable)

  @mock.patch.object(
      event_multiplexer.EventMultiplexer,
      "PluginRunToTagToContent",
      return_value={"bar": {
          "foo": "".encode("utf-8")
      }},
  )
  def testIsActive(self, get_random_stub):
    self.assertTrue(self._plugin.is_active())

  @mock.patch.object(
      event_multiplexer.EventMultiplexer,
      "PluginRunToTagToContent",
      return_value={})
  def testIsInactive(self, get_random_stub):
    self.assertFalse(self._plugin.is_active())

  def testIndexJsRoute(self):
    """Tests that the /tags route offers the correct run to tag mapping."""
    response = self._server.get("/data/plugin/fairness_indicators/index.js")
    self.assertEqual(200, response.status_code)

  def testVulcanizedTemplateRoute(self):
    """Tests that the /tags route offers the correct run to tag mapping."""
    response = self._server.get(
        "/data/plugin/fairness_indicators/vulcanized_tfma.js")
    self.assertEqual(200, response.status_code)

  def testGetEvalResultsRoute(self):
    model_location = self._exportEvalSavedModel(
        linear_classifier.simple_linear_classifier)
    examples = [
        self._makeExample(age=3.0, language="english", label=1.0),
        self._makeExample(age=3.0, language="chinese", label=0.0),
        self._makeExample(age=4.0, language="english", label=1.0),
        self._makeExample(age=5.0, language="chinese", label=1.0),
        self._makeExample(age=5.0, language="hindi", label=1.0)
    ]
    data_location = self._writeTFExamplesToTFRecords(examples)
    eval_config = tfma.EvalConfig(
        input_data_specs=[tfma.InputDataSpec(location=data_location)],
        model_specs=[tfma.ModelSpec(location=model_location)],
        output_data_specs=[
            tfma.OutputDataSpec(default_location=self._eval_result_output_dir)
        ])
    _ = tfma.run_model_analysis(
        eval_config=eval_config,
        eval_shared_model=tfma.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key="age"))

    response = self._server.get(
        "/data/plugin/fairness_indicators/get_evaluation_result?run=.")
    self.assertEqual(200, response.status_code)


if __name__ == "__main__":
  tf.test.main()
