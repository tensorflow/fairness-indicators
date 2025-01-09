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
"""TensorBoard Fairnss Indicators plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Union

from absl import logging
from tensorboard_plugin_fairness_indicators import metadata
import six
import tensorflow as tf
import tensorflow_model_analysis as tfma
from werkzeug import wrappers

from google.protobuf import json_format
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin


_TEMPLATE_LOCATION = os.path.normpath(
    os.path.join(
        __file__, '../../'
        'tensorflow_model_analysis/static/vulcanized_tfma.js'))


def stringify_slice_key_value(
    slice_key: tfma.slicer.slicer_lib.SliceKeyType,
) -> str:
  """Stringifies a slice key value.

  The string representation of a SingletonSliceKeyType is "feature:value". This
  function returns value.

  When
  multiple columns / features are specified, the string representation of a
  SliceKeyType's value is "v1_X_v2_X_..." where v1, v2, ... are values. For
  example,
  ('gender, 'f'), ('age', 5) becomes f_X_5. If no columns / feature
  specified, return "Overall".

  Note that we do not perform special escaping for slice values that contain
  '_X_'. This stringified representation is meant to be human-readbale rather
  than a reversible encoding.

  The columns will be in the same order as in SliceKeyType. If they are
  generated using SingleSliceSpec.generate_slices, they will be in sorted order,
  ascending.

  Technically float values are not supported, but we don't check for them here.

  Args:
    slice_key: Slice key to stringify. The constituent SingletonSliceKeyTypes
      should be sorted in ascending order.

  Returns:
    String representation of the slice key's value.
  """
  if not slice_key:
    return 'Overall'

  # Since this is meant to be a human-readable string, we assume that the
  # feature values are valid UTF-8 strings (might not be true in cases where
  # people store serialised protos in the features for instance).
  # We need to call as_str_any to convert non-string (e.g. integer) values to
  # string first before converting to text.
  # We use u'{}' instead of '{}' here to avoid encoding a unicode character with
  # ascii codec.
  values = [
      '{}'.format(tf.compat.as_text(tf.compat.as_str_any(value)))
      for _, value in slice_key
  ]
  return '_X_'.join(values)


def _add_cross_slice_key_data(
    slice_key: tfma.slicer.slicer_lib.CrossSliceKeyType,
    metrics: tfma.view.view_types.MetricsByTextKey,
    data: list[Any],
):
  """Adds data for cross slice key.

  Baseline and comparison slice keys are joined by '__XX__'.
  Args:
    slice_key: Cross slice key.
    metrics: Metrics data for the cross slice key.
    data: List where UI data is to be appended.
  """
  baseline_key = slice_key[0]
  comparison_key = slice_key[1]
  stringify_slice_value = (
      stringify_slice_key_value(baseline_key)
      + '__XX__'
      + stringify_slice_key_value(comparison_key)
  )
  stringify_slice = (
      tfma.slicer.slicer_lib.stringify_slice_key(baseline_key)
      + '__XX__'
      + tfma.slicer.slicer_lib.stringify_slice_key(comparison_key)
  )
  data.append({
      'sliceValue': stringify_slice_value,
      'slice': stringify_slice,
      'metrics': metrics,
  })


def convert_slicing_metrics_to_ui_input(
    slicing_metrics: list[
        tuple[
            tfma.slicer.slicer_lib.SliceKeyOrCrossSliceKeyType,
            tfma.view.view_types.MetricsByOutputName,
        ]
    ],
    slicing_column: Union[str, None] = None,
    slicing_spec: Union[tfma.slicer.slicer_lib.SingleSliceSpec, None] = None,
    output_name: str = '',
    multi_class_key: str = '',
) -> Union[list[dict[str, Any]], None]:
  """Renders the Fairness Indicator view.

  Args:
    slicing_metrics: tfma.EvalResult.slicing_metrics.
    slicing_column: The slicing column to to filter results. If both
      slicing_column and slicing_spec are None, show all eval results.
    slicing_spec: The slicing spec to filter results. If both slicing_column and
      slicing_spec are None, show all eval results.
    output_name: The output name associated with metric (for multi-output
      models).
    multi_class_key: The multi-class key associated with metric (for multi-class
      models).

  Returns:
    A list of dicts for each slice, where each dict contains keys 'sliceValue',
    'slice', and 'metrics'.

  Raises:
    ValueError if no related eval result found or both slicing_column and
    slicing_spec are not None.
  """
  if slicing_column and slicing_spec:
    raise ValueError(
        'Only one of the "slicing_column" and "slicing_spec" parameters '
        'can be set.'
    )
  if slicing_column:
    slicing_spec = tfma.slicer.slicer_lib.SingleSliceSpec(
        columns=[slicing_column]
    )

  data = []
  for slice_key, metric_value in slicing_metrics:
    if (
        metric_value is not None
        and output_name in metric_value
        and multi_class_key in metric_value[output_name]
    ):
      metrics = metric_value[output_name][multi_class_key]
      # To add evaluation data for cross slice comparison.
      if tfma.slicer.slicer_lib.is_cross_slice_key(slice_key):
        _add_cross_slice_key_data(slice_key, metrics, data)
      # To add evaluation data for regular slices.
      elif (
          slicing_spec is None
          or not slice_key
          or slicing_spec.is_slice_applicable(slice_key)
      ):
        data.append({
            'sliceValue': stringify_slice_key_value(slice_key),
            'slice': tfma.slicer.slicer_lib.stringify_slice_key(slice_key),
            'metrics': metrics,
        })
  if not data:
    raise ValueError(
        'No eval result found for output_name:"%s" and '
        'multi_class_key:"%s" and slicing_column:"%s" and slicing_spec:"%s".'
        % (output_name, multi_class_key, slicing_column, slicing_spec)
    )
  return data


class FairnessIndicatorsPlugin(base_plugin.TBPlugin):
  """A plugin to visualize Fairness Indicators."""

  plugin_name = metadata.PLUGIN_NAME

  def __init__(self, context):
    """Instantiates plugin via TensorBoard core.

    Args:
      context: A base_plugin.TBContext instance. A magic container that
        TensorBoard uses to make objects available to the plugin.
    """
    self._multiplexer = context.multiplexer

  def get_plugin_apps(self):
    """Gets all routes offered by the plugin.

    This method is called by TensorBoard when retrieving all the
    routes offered by the plugin.

    Returns:
      A dictionary mapping URL path to route that handles it.
    """
    return {
        '/get_evaluation_result':
            self._get_evaluation_result,
        '/get_evaluation_result_from_remote_path':
            self._get_evaluation_result_from_remote_path,
        '/index.js':
            self._serve_js,
        '/vulcanized_tfma.js':
            self._serve_vulcanized_js,
    }

  def frontend_metadata(self):
    return base_plugin.FrontendMetadata(
        es_module_path='/index.js',
        disable_reload=False,
        tab_name='Fairness Indicators',
        remove_dom=False,
        element_name=None)

  def is_active(self):
    """Determines whether this plugin is active.

    This plugin is only active if TensorBoard sampled any summaries
    relevant to the plugin.

    Returns:
      Whether this plugin is active.
    """
    return bool(
        self._multiplexer.PluginRunToTagToContent(
            FairnessIndicatorsPlugin.plugin_name))

  # pytype: disable=wrong-arg-types
  @wrappers.Request.application
  def _serve_js(self, request):
    filepath = os.path.join(os.path.dirname(__file__), 'static', 'index.js')
    with open(filepath) as infile:
      contents = infile.read()
    return http_util.Respond(
        request, contents, content_type='application/javascript')

  @wrappers.Request.application
  def _serve_vulcanized_js(self, request):
    with open(_TEMPLATE_LOCATION) as infile:
      contents = infile.read()
    return http_util.Respond(
        request, contents, content_type='application/javascript')

  @wrappers.Request.application
  def _get_evaluation_result(self, request):
    run = request.args.get('run')
    try:
      run = six.ensure_text(run)
    except (UnicodeDecodeError, AttributeError):
      pass

    data = []
    try:
      eval_result_output_dir = six.ensure_text(
          self._multiplexer.Tensors(run, FairnessIndicatorsPlugin.plugin_name)
          [0].tensor_proto.string_val[0])
      eval_result = tfma.load_eval_result(output_path=eval_result_output_dir)
      # TODO(b/141283811): Allow users to choose different model output names
      # and class keys in case of multi-output and multi-class model.
      data = convert_slicing_metrics_to_ui_input(eval_result.slicing_metrics)
    except (KeyError, json_format.ParseError) as error:
      logging.info('Error while fetching evaluation data, %s', error)
    return http_util.Respond(request, data, content_type='application/json')

  def _get_output_file_format(self, evaluation_output_path):
    file_format = os.path.splitext(evaluation_output_path)[1]
    if file_format:
      return file_format[1:]

    return ''

  @wrappers.Request.application
  def _get_evaluation_result_from_remote_path(self, request):
    evaluation_output_path = request.args.get('evaluation_output_path')
    try:
      evaluation_output_path = six.ensure_text(evaluation_output_path)
    except (UnicodeDecodeError, AttributeError):
      pass
    try:
      eval_result = tfma.load_eval_result(
          os.path.dirname(evaluation_output_path),
          output_file_format=self._get_output_file_format(
              evaluation_output_path))
      data = convert_slicing_metrics_to_ui_input(eval_result.slicing_metrics)
    except (KeyError, json_format.ParseError) as error:
      logging.info('Error while fetching evaluation data, %s', error)
      data = []
    return http_util.Respond(request, data, content_type='application/json')
  # pytype: enable=wrong-arg-types
