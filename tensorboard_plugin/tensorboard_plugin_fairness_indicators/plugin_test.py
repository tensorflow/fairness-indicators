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

from collections import abc
import os
import shutil
from unittest import mock

from tensorboard_plugin_fairness_indicators import plugin
from tensorboard_plugin_fairness_indicators import summary_v2
import six
import tensorflow as tf2
from tensorflow.keras import layers
import tensorflow_model_analysis as tfma
from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from tensorboard.backend import application
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer
from tensorboard.plugins import base_plugin

tf = tf2


# Define keras based linear classifier.
def create_linear_classifier(model_dir):

  inputs = tf.keras.Input(shape=(2,))
  outputs = layers.Dense(1, activation="sigmoid")(inputs)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  model.compile(
      optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
  )

  tf.saved_model.save(model, model_dir)
  return model


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
    self._server = werkzeug_test.Client(wsgi_app, wrappers.Response)
    self._routes = self._plugin.get_plugin_apps()

  def tearDown(self):
    super(PluginTest, self).tearDown()
    shutil.rmtree(self._log_dir, ignore_errors=True)

  def _export_eval_saved_model(self):
    """Export the evaluation saved model."""
    temp_eval_export_dir = os.path.join(self.get_temp_dir(), "eval_export_dir")
    return create_linear_classifier(temp_eval_export_dir)

  def _write_tf_examples_to_tfrecords(self, examples):
    data_location = os.path.join(self.get_temp_dir(), "input_data.rio")
    with tf.io.TFRecordWriter(data_location) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
    return data_location

  def _make_tf_example(self, age, language, label):
    example = tf.train.Example()
    example.features.feature["age"].float_list.value[:] = [age]
    example.features.feature["language"].bytes_list.value[:] = [
        six.ensure_binary(language, "utf8")
    ]
    example.features.feature["label"].float_list.value[:] = [label]
    return example

  def testRoutes(self):
    self.assertIsInstance(self._routes["/get_evaluation_result"],
                          abc.Callable)
    self.assertIsInstance(
        self._routes["/get_evaluation_result_from_remote_path"],
        abc.Callable)
    self.assertIsInstance(self._routes["/index.js"], abc.Callable)
    self.assertIsInstance(self._routes["/vulcanized_tfma.js"],
                          abc.Callable)

  @mock.patch.object(
      event_multiplexer.EventMultiplexer,
      "PluginRunToTagToContent",
      return_value={"bar": {
          "foo": "".encode("utf-8")
      }},
  )
  def testIsActive(self):
    self.assertTrue(self._plugin.is_active())

  @mock.patch.object(
      event_multiplexer.EventMultiplexer,
      "PluginRunToTagToContent",
      return_value={})
  def testIsInactive(self):
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
    model_location = self._export_eval_saved_model()  # Call the method
    examples = [
        self._make_tf_example(age=3.0, language="english", label=1.0),
        self._make_tf_example(age=3.0, language="chinese", label=0.0),
        self._make_tf_example(age=4.0, language="english", label=1.0),
        self._make_tf_example(age=5.0, language="chinese", label=1.0),
        self._make_tf_example(age=5.0, language="hindi", label=1.0),
    ]
    data_location = self._write_tf_examples_to_tfrecords(examples)
    _ = tfma.run_model_analysis(
        eval_shared_model=tfma.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key="age"),
        data_location=data_location,
        output_path=self._eval_result_output_dir)

    response = self._server.get(
        "/data/plugin/fairness_indicators/get_evaluation_result?run=.")
    self.assertEqual(200, response.status_code)

  def testGetEvalResultsFromURLRoute(self):
    model_location = self._export_eval_saved_model()  # Call the method
    examples = [
        self._make_tf_example(age=3.0, language="english", label=1.0),
        self._make_tf_example(age=3.0, language="chinese", label=0.0),
        self._make_tf_example(age=4.0, language="english", label=1.0),
        self._make_tf_example(age=5.0, language="chinese", label=1.0),
        self._make_tf_example(age=5.0, language="hindi", label=1.0),
    ]
    data_location = self._write_tf_examples_to_tfrecords(examples)
    _ = tfma.run_model_analysis(
        eval_shared_model=tfma.default_eval_shared_model(
            eval_saved_model_path=model_location, example_weight_key="age"),
        data_location=data_location,
        output_path=self._eval_result_output_dir)

    response = self._server.get(
        "/data/plugin/fairness_indicators/"
        + "get_evaluation_result_from_remote_path?evaluation_output_path="
        + self._eval_result_output_dir
    )
    self.assertEqual(200, response.status_code)

  def test_get_output_file_format(self):
    evaluation_output_path = os.path.join(
        self._eval_result_output_dir, "eval_result.tfrecord"
    )
    self.assertEqual(
        self._plugin._get_output_file_format(evaluation_output_path),
        "tfrecord",
    )


if __name__ == "__main__":
  tf.test.main()
