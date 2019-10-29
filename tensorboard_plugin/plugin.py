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
from __future__ import google_type_annotations
from __future__ import print_function

from absl import logging
import os

import google3
from fairness_indicators.tensorboard_plugin import metadata
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.addons.fairness.view import widget_view
from werkzeug import wrappers
from google3.net.proto2.python.public import json_format
from google3.third_party.tensorboard.backend import http_util
from google3.third_party.tensorboard.plugins import base_plugin

# TODO(b/141283811): Make vulcanized tfma file path portable to OSS.
_TEMPLATE_LOCATION = os.path.normpath(
    os.path.join(
        __file__, '../../../'
        'tensorflow_model_analysis/frontend/vulcanize/vulcanized_tfma.js'))


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
        '/get_evaluation_result': self._get_evaluation_result,
        '/index.js': self._serve_js,
        '/vulcanized_tfma.js': self._serve_vulcanized_js,
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
      run = run.decode()
    except (UnicodeDecodeError, AttributeError):
      pass
    try:
      eval_result_output_dir = self._multiplexer.Tensors(
          run, FairnessIndicatorsPlugin.plugin_name
      )[0].tensor_proto.string_val[0].decode('utf-8')
      eval_result = tfma.load_eval_result(output_path=eval_result_output_dir)
      # TODO(b/141283811): Allow users to choose different model output names
      # and class keys in case of multi-output and multi-class model.
      data = widget_view.convert_eval_result_to_ui_input(eval_result)
    except (KeyError, json_format.ParseError) as error:
      logging.info('Error while fetching evaluation data, %s', error)
      data = []
    return http_util.Respond(request, data, content_type='application/json')
