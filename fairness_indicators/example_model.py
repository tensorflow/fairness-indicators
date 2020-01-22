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

This scripts contains boilerplate code to train a DNNClassifier
and evaluate it using Tensorflow Model Analysis. Evaluation
results can be visualized using tools like TensorBoard.

Usage:

1. Train model:
  demo_script.train_model(...)

2. Evaluate:
  demo_script.evaluate_model(...)
"""

import os
import tempfile
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators  # pylint: disable=unused-import


def train_model(model_dir,
                train_tf_file,
                label,
                text_feature,
                feature_map,
                module_spec='https://tfhub.dev/google/nnlm-en-dim128/1'):
  """Train model using DNN Classifier.

  Args:
    model_dir: Directory path to save trained model.
    train_tf_file: File containing training TFRecordDataset.
    label: Groundtruth label.
    text_feature: Text feature to be evaluated.
    feature_map: Dict of feature names to their data type.
    module_spec: A module spec defining the module to instantiate or a path
      where to load a module spec.

  Returns:
    Trained DNNClassifier.
  """

  def train_input_fn():
    """Train Input function."""

    def parse_function(serialized):
      parsed_example = tf.io.parse_single_example(
          serialized=serialized, features=feature_map)
      # Adds a weight column to deal with unbalanced classes.
      parsed_example['weight'] = tf.add(parsed_example[label], 0.1)
      return (parsed_example, parsed_example[label])

    train_dataset = tf.data.TFRecordDataset(
        filenames=[train_tf_file]).map(parse_function).batch(512)
    return train_dataset

  text_embedding_column = hub.text_embedding_column(
      key=text_feature, module_spec=module_spec)

  classifier = tf.estimator.DNNClassifier(
      hidden_units=[500, 100],
      weight_column='weight',
      feature_columns=[text_embedding_column],
      n_classes=2,
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
      model_dir=model_dir)

  classifier.train(input_fn=train_input_fn, steps=1000)
  return classifier


def evaluate_model(classifier, validate_tf_file, tfma_eval_result_path,
                   selected_slice, label, feature_map):
  """Evaluate Model using Tensorflow Model Analysis.

  Args:
    classifier: Trained classifier model to be evaluted.
    validate_tf_file: File containing validation TFRecordDataset.
    tfma_eval_result_path: Directory path where eval results will be written.
    selected_slice: Feature for slicing the data.
    label: Groundtruth label.
    feature_map: Dict of feature names to their data type.
  """

  def eval_input_receiver_fn():
    """Eval Input Receiver function."""
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_placeholder')

    receiver_tensors = {'examples': serialized_tf_example}

    features = tf.io.parse_example(serialized_tf_example, feature_map)
    features['weight'] = tf.ones_like(features[label])

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=features[label])

  tfma_export_dir = tfma.export.export_eval_savedmodel(
      estimator=classifier,
      export_dir_base=os.path.join(tempfile.gettempdir(), 'tfma_eval_model'),
      eval_input_receiver_fn=eval_input_receiver_fn)

  # Define slices that you want the evaluation to run on.
  slice_spec = [
      tfma.slicer.SingleSliceSpec(),  # Overall slice
      tfma.slicer.SingleSliceSpec(columns=[selected_slice]),
  ]

  # Add the fairness metrics.
  # pytype: disable=module-attr
  add_metrics_callbacks = [
      tfma.post_export_metrics.fairness_indicators(
          thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], labels_key=label)
  ]
  # pytype: enable=module-attr

  eval_shared_model = tfma.default_eval_shared_model(
      eval_saved_model_path=tfma_export_dir,
      add_metrics_callbacks=add_metrics_callbacks)

  # Run the fairness evaluation.
  tfma.run_model_analysis(
      eval_shared_model=eval_shared_model,
      data_location=validate_tf_file,
      output_path=tfma_eval_result_path,
      slice_spec=slice_spec)
