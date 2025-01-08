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
"""Demo script to train and evaluate a model.

This scripts contains boilerplate code to train a Keras Text Classifier
and evaluate it using Tensorflow Model Analysis. Evaluation
results can be visualized using tools like TensorBoard.
"""

from typing import Any

from fairness_indicators import fairness_indicators_metrics  # pylint: disable=unused-import
from tensorflow import keras
import tensorflow.compat.v1 as tf
import tensorflow_model_analysis as tfma


TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'
SLICE = 'slice'
FEATURE_MAP = {
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    TEXT_FEATURE: tf.io.FixedLenFeature([], tf.string),
    SLICE: tf.io.VarLenFeature(tf.string),
}


class ExampleParser(keras.layers.Layer):
  """A Keras layer that parses the tf.Example."""

  def __init__(self, input_feature_key):
    self._input_feature_key = input_feature_key
    self.input_spec = keras.layers.InputSpec(shape=(1,), dtype=tf.string)
    super().__init__()

  def compute_output_shape(self, input_shape: Any):
    return [1, 1]

  def call(self, serialized_examples):
    def get_feature(serialized_example):
      parsed_example = tf.io.parse_single_example(
          serialized_example, features=FEATURE_MAP
      )
      return parsed_example[self._input_feature_key]
    serialized_examples = tf.cast(serialized_examples, tf.string)
    return tf.map_fn(get_feature, serialized_examples)


class Reshaper(keras.layers.Layer):
  """A Keras layer that reshapes the input."""

  def call(self, inputs):
    return tf.reshape(inputs, (1, 32))


class Caster(keras.layers.Layer):
  """A Keras layer that reshapes the input."""

  def call(self, inputs):
    return tf.cast(inputs, tf.float32)


def get_example_model(input_feature_key: str):
  """Returns a Keras model for testing."""
  parser = ExampleParser(input_feature_key)
  text_vectorization = keras.layers.TextVectorization(
      max_tokens=32,
      output_mode='int',
      output_sequence_length=32,
  )
  text_vectorization.adapt(
      ['nontoxic', 'toxic comment', 'test comment', 'abc', 'abcdef', 'random']
  )
  dense1 = keras.layers.Dense(
      32,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
  )
  dense2 = keras.layers.Dense(
      1,
      activation=None,
      use_bias=False,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
  )

  inputs = tf.keras.Input(shape=(), dtype=tf.string)
  parsed_example = parser(inputs)
  text_vector = text_vectorization(parsed_example)
  text_vector = Reshaper()(text_vector)
  text_vector = Caster()(text_vector)
  output1 = dense1(text_vector)
  output2 = dense2(output1)
  return tf.keras.Model(inputs=inputs, outputs=output2)


def evaluate_model(
    classifier_model_path,
    validate_tf_file_path,
    tfma_eval_result_path,
    eval_config,
):
  """Evaluate Model using Tensorflow Model Analysis.

  Args:
    classifier_model_path: Trained classifier model to be evaluted.
    validate_tf_file_path: File containing validation TFRecordDataset.
    tfma_eval_result_path: Path to export tfma-related eval path.
    eval_config: tfma eval_config.
  """

  eval_shared_model = tfma.default_eval_shared_model(
      eval_saved_model_path=classifier_model_path, eval_config=eval_config
  )

  # Run the fairness evaluation.
  tfma.run_model_analysis(
      eval_shared_model=eval_shared_model,
      data_location=validate_tf_file_path,
      output_path=tfma_eval_result_path,
      eval_config=eval_config,
  )
