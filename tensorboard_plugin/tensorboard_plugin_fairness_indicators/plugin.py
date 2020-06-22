# Lint as: python2, python3
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

# Standard imports
from absl import logging
from tensorboard_plugin_fairness_indicators import metadata
import six
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.addons.fairness.view import widget_view
from werkzeug import wrappers
from google.protobuf import json_format
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin

_TEMPLATE_LOCATION = os.path.normpath(
    os.path.join(
        __file__, '../../'
        'tensorflow_model_analysis/static/vulcanized_tfma.js'))


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
      data = widget_view.convert_slicing_metrics_to_ui_input(
          eval_result.slicing_metrics)
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
      data = widget_view.convert_slicing_metrics_to_ui_input(
          eval_result.slicing_metrics)
    except (KeyError, json_format.ParseError) as error:
      logging.info('Error while fetching evaluation data, %s', error)
      data = []
    return http_util.Respond(request, data, content_type='application/json')
