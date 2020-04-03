# Lint as: python3
"""Tests for fairness_indicators.remediation.weight_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Standard imports
from fairness_indicators.remediation import weight_utils
import tensorflow.compat.v1 as tf


EvalResult = collections.namedtuple('EvalResult', ['slicing_metrics'])


class WeightUtilsTest(tf.test.TestCase):

  def create_eval_result(self):
    return EvalResult(slicing_metrics=[
        ((), {
            '': {
                '': {
                    'post_export_metrics/negative_rate@0.10': {
                        'doubleValue': 0.08
                    },
                    'accuracy': {
                        'doubleValue': 0.444
                    }
                }
            }
        }),
        ((('gender', 'female'),), {
            '': {
                '': {
                    'post_export_metrics/negative_rate@0.10': {
                        'doubleValue': 0.09
                    },
                    'accuracy': {
                        'doubleValue': 0.333
                    }
                }
            }
        }),
        (((u'gender', u'female'),
          (u'sexual_orientation', u'homosexual_gay_or_lesbian')), {
              '': {
                  '': {
                      'post_export_metrics/negative_rate@0.10': {
                          'doubleValue': 0.1
                      },
                      'accuracy': {
                          'doubleValue': 0.222
                      }
                  }
              }
          }),
    ])

  def create_bounded_result(self):
    return EvalResult(slicing_metrics=[
        ((), {
            '': {
                '': {
                    'post_export_metrics/negative_rate@0.10': {
                        'boundedValue': {
                            'lowerBound': 0.07,
                            'upperBound': 0.09,
                            'value': 0.08,
                            'methodology': 'POISSON_BOOTSTRAP'
                        }
                    },
                    'accuracy': {
                        'boundedValue': {
                            'lowerBound': 0.07,
                            'upperBound': 0.09,
                            'value': 0.444,
                            'methodology': 'POISSON_BOOTSTRAP'
                        }
                    }
                }
            }
        }),
        ((('gender', 'female'),), {
            '': {
                '': {
                    'post_export_metrics/negative_rate@0.10': {
                        'boundedValue': {
                            'lowerBound': 0.07,
                            'upperBound': 0.09,
                            'value': 0.09,
                            'methodology': 'POISSON_BOOTSTRAP'
                        }
                    },
                    'accuracy': {
                        'boundedValue': {
                            'lowerBound': 0.07,
                            'upperBound': 0.09,
                            'value': 0.333,
                            'methodology': 'POISSON_BOOTSTRAP'
                        }
                    }
                }
            }
        }),
        (((u'gender', u'female'),
          (u'sexual_orientation', u'homosexual_gay_or_lesbian')), {
              '': {
                  '': {
                      'post_export_metrics/negative_rate@0.10': {
                          'boundedValue': {
                              'lowerBound': 0.07,
                              'upperBound': 0.09,
                              'value': 0.1,
                              'methodology': 'POISSON_BOOTSTRAP'
                          }
                      },
                      'accuracy': {
                          'boundedValue': {
                              'lowerBound': 0.07,
                              'upperBound': 0.09,
                              'value': 0.222,
                              'methodology': 'POISSON_BOOTSTRAP'
                          }
                      }
                  }
              }
          }),
    ])

  def test_baseline(self):
    test_eval_result = self.create_eval_result()
    self.assertEqual(
        0.08,
        weight_utils.get_baseline_value(
            test_eval_result, 'Overall',
            'post_export_metrics/negative_rate@0.10'))
    self.assertEqual(
        0.09,
        weight_utils.get_baseline_value(
            test_eval_result, (('gender', 'female'),),
            'post_export_metrics/negative_rate@0.10'))
    # Test 'accuracy'.
    self.assertEqual(
        0.444,
        weight_utils.get_baseline_value(test_eval_result, 'Overall',
                                        'accuracy'))
    # Test intersectional metrics.
    self.assertEqual(
        0.222,
        weight_utils.get_baseline_value(
            test_eval_result,
            ((u'gender', u'female'),
             (u'sexual_orientation', u'homosexual_gay_or_lesbian')),
            'accuracy'))
    with self.assertRaises(ValueError):
      # Test slice not found.
      weight_utils.get_baseline_value(test_eval_result,
                                      (('nonexistant', 'slice'),), 'accuracy')
    with self.assertRaises(KeyError):
      # Test metric not found.
      weight_utils.get_baseline_value(test_eval_result, (('gender', 'female'),),
                                      'nonexistent_metric')

  def test_get_metric_value_raise_key_error(self):
    input_dict = {'': {'': {'accuracy': 0.1}}}
    metric_name = 'nonexistent_metric'
    with self.assertRaises(KeyError):
      weight_utils._get_metric_value(input_dict, metric_name)

  def test_get_metric_value_raise_unsupported_value(self):
    input_dict = {
        '': {
            '': {
                'accuracy': {
                    'boundedValue': {1}
                }
            }
        }
    }
    metric_name = 'accuracy'
    with self.assertRaises(TypeError):
      weight_utils._get_metric_value(input_dict, metric_name)

  def test_get_metric_value_raise_empty_dict(self):
    with self.assertRaises(KeyError):
      weight_utils._get_metric_value({}, 'metric_name')

  def test_create_difference_dictionary(self):
    test_eval_result = self.create_eval_result()
    res = weight_utils.create_percentage_difference_dictionary(
        test_eval_result, 'Overall', 'post_export_metrics/negative_rate@0.10')
    self.assertEqual(3, len(res))
    self.assertIn('gender-sexual_orientation', res)
    self.assertIn('gender', res)
    self.assertAlmostEqual(res['gender']['female'], 0.125)
    self.assertAlmostEqual(res[''][''], 0)

  def test_create_difference_dictionary_baseline(self):
    test_eval_result = self.create_eval_result()
    res = weight_utils.create_percentage_difference_dictionary(
        test_eval_result, (('gender', 'female'),),
        'post_export_metrics/negative_rate@0.10')
    self.assertEqual(3, len(res))
    self.assertIn('gender-sexual_orientation', res)
    self.assertIn('gender', res)
    self.assertAlmostEqual(res['gender']['female'], 0)
    self.assertAlmostEqual(res[''][''], -0.11111111)

  def test_create_difference_dictionary_bounded_metrics(self):
    test_eval_result = self.create_bounded_result()
    res = weight_utils.create_percentage_difference_dictionary(
        test_eval_result, 'Overall', 'post_export_metrics/negative_rate@0.10')
    self.assertEqual(3, len(res))
    self.assertIn('gender-sexual_orientation', res)
    self.assertIn('gender', res)
    self.assertAlmostEqual(res['gender']['female'], 0.125)
    self.assertAlmostEqual(res[''][''], 0)


if __name__ == '__main__':
  tf.test.main()
