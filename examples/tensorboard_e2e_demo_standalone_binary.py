"""Fairness Indicators TensorBoard plugin demo script.

Using this script, you can visualize realtime training metrics and plots
along with fairness evaluation results (under Fairness Indicators tab) in
TensorBoard.

Note: This script does not run the plugin, but performs necessary steps to
enable a demo of the Fairness Indicators TensorBoard plugin

Please run following steps for end-to-end demo. This will start a local
TensorBoard instance. After the local instance is started, a link will be
displayed to the terminal. Open the link in your browser to view the outcome.

Step 1: Install all necessary pip packages
  python3 -m virtualenv ~/tensorboard_demo
  source ~/tensorboard_demo/bin/activate
  pip install --upgrade pip
  pip install fairness-indicators
  pip install tensorflow_hub
  pip install tensorboard_plugin_fairness_indicators
  pip uninstall -y tensorboard
  pip install --upgrade tb-nightly

Step 2: Start TensorBoard
  tensorboard --logdir=/tmp/train/

Step 3: Run the python binary
  python ./tensorboard_e2e_demo_standalone_binary.py
"""

import datetime
import os
import tempfile
import apache_beam as beam
from tensorboard_plugin_fairness_indicators import summary_v2
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators  # pylint: disable=unused-import

tf.compat.v1.enable_eager_execution()

################################################################################
# Download training and validation data.
train_tf_file = tf.keras.utils.get_file(
    'train.tf',
    'https://storage.googleapis.com/civil_comments_dataset/train.tfrecord')
validate_tf_file = tf.keras.utils.get_file(
    'validate.tf',
    'https://storage.googleapis.com/civil_comments_dataset/validate.tfrecord')

################################################################################
# Define Constants
BASE_DIR = tempfile.gettempdir()
TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'
FEATURE_MAP = {
    # Label:
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    # Text:
    TEXT_FEATURE: tf.io.FixedLenFeature([], tf.string),

    # Identities:
    'gender': tf.io.VarLenFeature(tf.string),
}


################################################################################
# Train the Model
def train_input_fn():
  """Training input function."""

  def parse_function(serialized):
    parsed_example = tf.io.parse_single_example(
        serialized=serialized, features=FEATURE_MAP)
    # Toxic example are ~10x of non-toxic examples, so assigning them weight
    # accordingly to deal with unbalanced data.
    parsed_example['weight'] = tf.add(parsed_example[LABEL], 0.1)
    return (parsed_example, parsed_example[LABEL])

  train_dataset = tf.data.TFRecordDataset(
      filenames=[train_tf_file]).map(parse_function).batch(512)
  return train_dataset


model_dir = os.path.join(BASE_DIR, 'train',
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

embedded_text_feature_column = hub.text_embedding_column(
    key=TEXT_FEATURE, module_spec='https://tfhub.dev/google/nnlm-en-dim128/1')

classifier = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    weight_column='weight',
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.optimizers.Adam(learning_rate=0.003),
    model_dir=model_dir)

classifier.train(input_fn=train_input_fn, steps=1000)


################################################################################
# Evaluate the model
def eval_input_receiver_fn():
  """Eval Input Receiver Function."""
  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_placeholder')

  # This *must* be a dictionary containing a single key 'examples', which
  # points to the input placeholder.
  receiver_tensors = {'examples': serialized_tf_example}

  features = tf.io.parse_example(serialized_tf_example, FEATURE_MAP)
  features['weight'] = tf.ones_like(features[LABEL])

  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors,
      labels=features[LABEL])


tfma_export_dir = tfma.export.export_eval_savedmodel(
    estimator=classifier,
    export_dir_base=os.path.join(BASE_DIR, 'tfma_eval_model'),
    eval_input_receiver_fn=eval_input_receiver_fn)

tfma_eval_result_path = os.path.join(BASE_DIR, 'tfma_eval_result')

slice_spec = [
    tfma.slicer.SingleSliceSpec(),  # Overall slice
    tfma.slicer.SingleSliceSpec(columns=['gender']),
]

add_metrics_callbacks = [
    tfma.post_export_metrics.fairness_indicators(
        thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], labels_key=LABEL)
]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=add_metrics_callbacks)

validate_dataset = tf.data.TFRecordDataset(filenames=[validate_tf_file])

with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create([v.numpy() for v in validate_dataset])
      | 'ExtractEvaluateAndWriteResults' >> tfma.ExtractEvaluateAndWriteResults(
          eval_shared_model=eval_shared_model,
          slice_spec=slice_spec,
          output_path=tfma_eval_result_path))

################################################################################
# Write fairness indicators evaluation results summary.
writer = tf.summary.create_file_writer(
    os.path.join(model_dir, 'fairness_indicators'))
with writer.as_default():
  summary_v2.FairnessIndicators(tfma_eval_result_path, step=1)
writer.close()
