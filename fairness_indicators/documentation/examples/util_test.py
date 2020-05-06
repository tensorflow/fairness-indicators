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
"""Tests for fairness_indicators.examples.util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from fairness_indicators.examples import util
import tensorflow.compat.v1 as tf

from google.protobuf import text_format


class UtilTest(tf.test.TestCase):

  def _create_example(self):
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
    filename = os.path.join(tempfile.mkdtemp(), 'input')
    with tf.io.TFRecordWriter(filename) as writer:
      for e in examples:
        writer.write(e.SerializeToString())
    return filename

  def test_convert_data(self):
    input_file = self._write_tf_records(self._create_example())
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


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
