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
"""Util methods for the example colabs."""

import os
import os.path
import tempfile

import pandas as pd
import tensorflow as tf

TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'

SEXUAL_ORIENTATION_COLUMNS = [
    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
    'other_sexual_orientation'
]
GENDER_COLUMNS = ['male', 'female', 'transgender', 'other_gender']
RELIGION_COLUMNS = [
    'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist',
    'other_religion'
]
RACE_COLUMNS = ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
DISABILITY_COLUMNS = [
    'physical_disability', 'intellectual_or_learning_disability',
    'psychiatric_or_mental_illness', 'other_disability'
]

IDENTITY_COLUMNS = {
    'gender': GENDER_COLUMNS,
    'sexual_orientation': SEXUAL_ORIENTATION_COLUMNS,
    'religion': RELIGION_COLUMNS,
    'race': RACE_COLUMNS,
    'disability': DISABILITY_COLUMNS
}

_THRESHOLD = 0.5


def convert_comments_data(input_filename, output_filename=None):
  """Convert the public civil comments data.

  In the orginal dataset
  https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
  for each indentity annotation columns, the value comes
  from percent of raters thought the comment referenced the identity. When
  processing the raw data, the threshold 0.5 is chosen and the identity terms
  are grouped together by their categories. For example if one comment has {
  male: 0.3, female: 1.0, transgender: 0.0, heterosexual: 0.8,
  homosexual_gay_or_lesbian: 1.0 }. After the processing, the data will be {
  gender: [female], sexual_orientation: [heterosexual,
  homosexual_gay_or_lesbian] }.

  Args:
    input_filename: The path to the raw civil comments data, with extension
      'tfrecord' or 'csv'.
    output_filename: The path to write the processed civil comments data.

  Returns:
    The file path to the converted dataset.

  Raises:
    ValueError: If the input_filename does not have a supported extension.
  """
  extension = os.path.splitext(input_filename)[1][1:]

  if not output_filename:
    output_filename = os.path.join(tempfile.mkdtemp(), 'output.' + extension)

  if extension == 'tfrecord':
    return _convert_comments_data_tfrecord(input_filename, output_filename)
  elif extension == 'csv':
    return _convert_comments_data_csv(input_filename, output_filename)

  raise ValueError(
      'input_filename must have supported file extension csv or tfrecord, '
      'given: {}'.format(input_filename))


def _convert_comments_data_tfrecord(input_filename, output_filename=None):
  """Convert the public civil comments data, for tfrecord data."""
  with tf.io.TFRecordWriter(output_filename) as writer:
    for serialized in tf.data.TFRecordDataset(filenames=[input_filename]):
      example = tf.train.Example()
      example.ParseFromString(serialized.numpy())
      if not example.features.feature[TEXT_FEATURE].bytes_list.value:
        continue

      new_example = tf.train.Example()
      new_example.features.feature[TEXT_FEATURE].bytes_list.value.extend(
          example.features.feature[TEXT_FEATURE].bytes_list.value)
      new_example.features.feature[LABEL].float_list.value.append(
          1 if example.features.feature[LABEL].float_list.value[0] >= _THRESHOLD
          else 0)

      for identity_category, identity_list in IDENTITY_COLUMNS.items():
        grouped_identity = []
        for identity in identity_list:
          if (example.features.feature[identity].float_list.value and
              example.features.feature[identity].float_list.value[0] >=
              _THRESHOLD):
            grouped_identity.append(identity.encode())
        new_example.features.feature[identity_category].bytes_list.value.extend(
            grouped_identity)
      writer.write(new_example.SerializeToString())

  return output_filename


def _convert_comments_data_csv(input_filename, output_filename=None):
  """Convert the public civil comments data, for csv data."""
  df = pd.read_csv(input_filename)

  # Filter out rows with empty comment text values.
  df = df[df[TEXT_FEATURE].ne('')]
  df = df[df[TEXT_FEATURE].notnull()]

  new_df = pd.DataFrame()
  new_df[TEXT_FEATURE] = df[TEXT_FEATURE]

  # Reduce the label to value 0 or 1.
  new_df[LABEL] = df[LABEL].ge(_THRESHOLD).astype(int)

  # Extract the list of all identity terms that exceed the threshold.
  def identity_conditions(df, identity_list):
    group = []
    for identity in identity_list:
      if df[identity] >= _THRESHOLD:
        group.append(identity)
    return group

  for identity_category, identity_list in IDENTITY_COLUMNS.items():
    new_df[identity_category] = df.apply(
        identity_conditions, args=((identity_list),), axis=1)

  new_df.to_csv(
      output_filename,
      header=[TEXT_FEATURE, LABEL, *IDENTITY_COLUMNS.keys()],
      index=False)

  return output_filename
