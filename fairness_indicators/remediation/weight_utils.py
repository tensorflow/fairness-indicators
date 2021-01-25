# Lint as: python2, python3
"""Utilities to suggest weights based on model analysis results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Mapping, Text

import tensorflow_model_analysis as tfma


def create_percentage_difference_dictionary(
    eval_result: tfma.EvalResult,
    baseline_name: Text, metric_name: Text) -> Dict[Text, Any]:
  """Creates dictionary of a % difference between a baseline and other slices.

  Args:
    eval_result: Loaded eval result from running TensorFlow Model Analysis.
    baseline_name: Name of the baseline slice, 'Overall' or a specified tuple.
    metric_name: Name of the metric on which to perform comparisons.

  Returns:
    Dictionary mapping slices to percentage difference from the baseline slice.
  """
  baseline_value = get_baseline_value(eval_result, baseline_name, metric_name)
  difference = {}
  for metrics_tuple in eval_result.slicing_metrics:
    slice_key = metrics_tuple[0]
    metrics = metrics_tuple[1]
    # Concatenate feature name/values for intersectional features.
    column = '-'.join([elem[0] for elem in slice_key])
    feature_val = '-'.join([elem[1] for elem in slice_key])
    if column not in difference:
      difference[column] = {}
    difference[column][feature_val] = (_get_metric_value(metrics, metric_name)
                                       - baseline_value) / baseline_value
  return difference


def _get_metric_value(
    nested_dict: Mapping[Text, Mapping[Text, Any]], metric_name: Text) -> float:
  """Returns the value of the named metric from a slice's metrics.

  Args:
    nested_dict: Dictionary of metrics from slice.
    metric_name: Value to return from the metric slice.

  Returns:
    Percentage value of the baseline slice name requested.

  Raises:
    KeyError: If the metric name isn't found in the metrics dictionary or if the
      input metrics dictionary is empty.
    TypeError: If an unsupported value type is found within dictionary slice.
      passed.
  """
  for value in nested_dict.values():
    if metric_name in value['']:
      typed_value = value[''][metric_name]
      if 'doubleValue' in typed_value:
        return typed_value['doubleValue']
      if 'boundedValue' in typed_value:
        return typed_value['boundedValue']['value']
      raise TypeError('Unsupported value type: %s' % typed_value)
    else:
      raise KeyError('Key %s not found in %s' %
                     (metric_name, list(value[''].keys())))
  raise KeyError(
      'Unable to return a metric value because the dictionary passed is empty.')


def get_baseline_value(
    eval_result: tfma.EvalResult,
    baseline_name: Text, metric_name: Text) -> float:
  """Looks through the evaluation result for the value of the baseline slice.

  Args:
    eval_result: Loaded eval result from running TensorFlow Model Analysis.
    baseline_name: Name of the baseline slice, 'Overall' or a specified tuple.
    metric_name: Name of the metric on which to perform comparisons.

  Returns:
    Percentage value of the baseline slice name requested.

  Raises:
    Value error if the baseline slice is not found in eval_results.
  """
  for metrics_tuple in eval_result.slicing_metrics:
    slice_tuple = metrics_tuple[0]
    if baseline_name == 'Overall' and not slice_tuple:
      return _get_metric_value(metrics_tuple[1], metric_name)
    if baseline_name == slice_tuple:
      return _get_metric_value(metrics_tuple[1], metric_name)
  raise ValueError('Could not find baseline %s in eval_result: %s' %
                   (baseline_name, eval_result))
