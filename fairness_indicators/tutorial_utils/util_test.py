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
"""Tests for fairness_indicators.tutorial_utils.util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import tempfile
from unittest import mock
from fairness_indicators.tutorial_utils import util
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format


class UtilTest(tf.test.TestCase):

  def _create_example_tfrecord(self):
    example = text_format.Parse(
        """
        features {
          feature { key: "comment_text"
                    value { bytes_list { value: [ "comment 1" ] }}
                  }
          feature { key: "toxicity" value { float_list { value: [ 0.1 ] }}}

          feature { key: "heterosexual" value { float_list { value: [ 0.1 ] }}}
          feature { key: "homosexual_gay_or_lesbian"
                    value { float_list { value: [ 0.1 ] }}
                  }
          feature { key: "bisexual" value { float_list { value: [ 0.5 ] }}}
          feature { key: "other_sexual_orientation"
                    value { float_list { value: [ 0.1 ] }}
                  }

          feature { key: "male" value { float_list { value: [ 0.1 ] }}}
          feature { key: "female" value { float_list { value: [ 0.2 ] }}}
          feature { key: "transgender" value { float_list { value: [ 0.3 ] }}}
          feature { key: "other_gender" value { float_list { value: [ 0.4 ] }}}

          feature { key: "christian" value { float_list { value: [ 0.0 ] }}}
          feature { key: "jewish" value { float_list { value: [ 0.1 ] }}}
          feature { key: "muslim" value { float_list { value: [ 0.2 ] }}}
          feature { key: "hindu" value { float_list { value: [ 0.3 ] }}}
          feature { key: "buddhist" value { float_list { value: [ 0.4 ] }}}
          feature { key: "atheist" value { float_list { value: [ 0.5 ] }}}
          feature { key: "other_religion"
                    value { float_list { value: [ 0.6 ] }}
                  }

          feature { key: "black" value { float_list { value: [ 0.1 ] }}}
          feature { key: "white" value { float_list { value: [ 0.2 ] }}}
          feature { key: "asian" value { float_list { value: [ 0.3 ] }}}
          feature { key: "latino" value { float_list { value: [ 0.4 ] }}}
          feature { key: "other_race_or_ethnicity"
                    value { float_list { value: [ 0.5 ] }}
                  }

          feature { key: "physical_disability"
                    value { float_list { value: [ 0.6 ] }}
                  }
          feature { key: "intellectual_or_learning_disability"
                    value { float_list { value: [ 0.7 ] }}
                  }
          feature { key: "psychiatric_or_mental_illness"
                    value { float_list { value: [ 0.8 ] }}
                  }
          feature { key: "other_disability"
                    value { float_list { value: [ 1.0 ] }}
                  }
        }
        """, tf.train.Example())
    empty_comment_example = text_format.Parse(
        """
        features {
          feature { key: "comment_text"
                    value { bytes_list {} }
                  }
          feature { key: "toxicity" value { float_list { value: [ 0.1 ] }}}
        }
        """, tf.train.Example())
    return [example, empty_comment_example]

  def _write_tf_records(self, examples):
    filename = os.path.join(tempfile.mkdtemp(), 'input.tfrecord')
    with tf.io.TFRecordWriter(filename) as writer:
      for e in examples:
        writer.write(e.SerializeToString())
    return filename

  def test_convert_data_tfrecord(self):
    input_file = self._write_tf_records(self._create_example_tfrecord())
    output_file = util.convert_comments_data(input_file)
    output_example_list = []
    for serialized in tf.data.TFRecordDataset(filenames=[output_file]):
      output_example = tf.train.Example()
      output_example.ParseFromString(serialized.numpy())
      output_example_list.append(output_example)

    self.assertEqual(len(output_example_list), 1)
    self.assertEqual(
        output_example_list[0],
        text_format.Parse(
            """
        features {
          feature { key: "comment_text"
                    value { bytes_list {value: [ "comment 1" ] }}
                  }
          feature { key: "toxicity" value { float_list { value: [ 0.0 ] }}}
          feature { key: "sexual_orientation"
                    value { bytes_list { value: ["bisexual"] }}
                  }
          feature { key: "gender" value { bytes_list { }}}
          feature { key: "race"
                    value { bytes_list { value: [ "other_race_or_ethnicity" ] }}
                  }
          feature { key: "religion"
                    value { bytes_list {
                      value: [  "atheist", "other_religion" ] }
                    }
                  }
          feature { key: "disability" value { bytes_list {
                    value: [
                      "physical_disability",
                      "intellectual_or_learning_disability",
                      "psychiatric_or_mental_illness",
                      "other_disability"] }}
                  }
        }
        """, tf.train.Example()))

  def _create_example_csv(self, use_fake_embedding=False):
    header = [
        'comment_text',
        'toxicity',
        'heterosexual',
        'homosexual_gay_or_lesbian',
        'bisexual',
        'other_sexual_orientation',
        'male',
        'female',
        'transgender',
        'other_gender',
        'christian',
        'jewish',
        'muslim',
        'hindu',
        'buddhist',
        'atheist',
        'other_religion',
        'black',
        'white',
        'asian',
        'latino',
        'other_race_or_ethnicity',
        'physical_disability',
        'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness',
        'other_disability',
    ]
    example = [
        'comment 1' if not use_fake_embedding else 0.35,
        0.1,
        # sexual orientation
        0.1,
        0.1,
        0.5,
        0.1,
        # gender
        0.1,
        0.2,
        0.3,
        0.4,
        # religion
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        # race or ethnicity
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        # disability
        0.6,
        0.7,
        0.8,
        1.0,
    ]
    empty_comment_example = [
        '' if not use_fake_embedding else 0.35,
        0.1,
        0.1,
        0.1,
        0.5,
        0.1,
        0.1,
        0.2,
        0.3,
        0.4,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
    ]
    return [header, example, empty_comment_example]

  def _write_csv(self, examples):
    filename = os.path.join(tempfile.mkdtemp(), 'input.csv')
    with open(filename, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile, delimiter=',')
      for example in examples:
        csvwriter.writerow(example)

    return filename

  def test_convert_data_csv(self):
    input_file = self._write_csv(self._create_example_csv())
    output_file = util.convert_comments_data(input_file)

    # Remove the quotes around identity terms list that read_csv injects.
    df = pd.read_csv(output_file).replace("'", '', regex=True)

    expected_df = pd.DataFrame()
    expected_df = expected_df.append(
        {
            'comment_text':
                'comment 1',
            'toxicity':
                0.0,
            'gender': [],
            'sexual_orientation': ['bisexual'],
            'race': ['other_race_or_ethnicity'],
            'religion': ['atheist', 'other_religion'],
            'disability': [
                'physical_disability', 'intellectual_or_learning_disability',
                'psychiatric_or_mental_illness', 'other_disability'
            ]
        },
        ignore_index=True)

    self.assertEqual(
        df.reset_index(drop=True, inplace=True),
        expected_df.reset_index(drop=True, inplace=True))

  # TODO(b/172260507): we should also look into testing the e2e call with tfma.
  @mock.patch(
      'tensorflow_model_analysis.default_eval_shared_model', autospec=True)
  @mock.patch('tensorflow_model_analysis.run_model_analysis', autospec=True)
  def test_get_eval_results_called_correclty(self, mock_run_model_analysis,
                                             mock_shared_model):
    mock_model = 'model'
    mock_shared_model.return_value = mock_model

    model_location = 'saved_model'
    eval_results_path = 'eval_results'
    data_file = 'data'
    util.get_eval_results(model_location, eval_results_path, data_file)

    mock_shared_model.assert_called_once_with(
        eval_saved_model_path=model_location, tags=[tf.saved_model.SERVING])

    expected_eval_config = text_format.Parse(
        """
     model_specs {
       label_key: 'toxicity'
     }
     metrics_specs {
       metrics {class_name: "AUC"}
       metrics {class_name: "ExampleCount"}
       metrics {class_name: "Accuracy"}
       metrics {
          class_name: "FairnessIndicators"
          config: '{"thresholds": [0.4, 0.4125, 0.425, 0.4375, 0.45, 0.4675, 0.475, 0.4875, 0.5]}'
       }
     }
     slicing_specs {
       feature_keys: 'religion'
     }
     slicing_specs {}
     options {
         compute_confidence_intervals { value: true }
         disabled_outputs{values: "analysis"}
     }
     """, tfma.EvalConfig())

    mock_run_model_analysis.assert_called_once_with(
        eval_shared_model=mock_model,
        data_location=data_file,
        file_format='tfrecords',
        eval_config=expected_eval_config,
        output_path=eval_results_path,
        extractors=None)

if __name__ == '__main__':
  tf.test.main()
