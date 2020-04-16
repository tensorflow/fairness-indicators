import os
import unittest
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


# TODO(karanshukla): names: _run_notebook and _write_notebook?
def _run_notebook(path):

  # TODO(karanshukla): can I comment this out?
  def _write_notebook(notebook):
    dirname = os.path.dirname(path)
    notebookname, _ = os.path.splitext(os.path.basename(path))
    output_path = os.path.join(dirname,
                               '{}_all_output.ipynb'.format(notebookname))
    with open(output_path, mode='wt') as f:
      nbformat.write(notebook, f)

  with open(path) as f:
    notebook = nbformat.read(f, as_version=4)

  proc = ExecutePreprocessor(timeout=None, kernel_name='python3')
  proc.allow_errors = True
  proc.preprocess(notebook, {'metadata': {'path': '/'}})

  _write_notebook(notebook)

  errors = []
  for cell in notebook.cells:
    if 'outputs' in cell:
      for output in cell['outputs']:
        if output.output_type == 'error':
          errors.append(output)

  return errors


class TestNotebook(unittest.TestCase):

  def test_example(self):
    errors = _run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_Example_Colab.ipynb')
    self.assertEqual(errors, [])


#   def test_facessd(self):
#     errors = _run_notebook(
#         'fairness_indicators/examples/Facessd_Fairness_Indicators_Example_Colab.ipynb')
#     self.assertEqual(errors, [])

#   def test_lineage(self):
#     errors = _run_notebook(
#         'fairness_indicators/examples/Fairness_Indicators_Lineage_Case_Study.ipynb')
#     self.assertEqual(errors, [])

#   def test_tb(self):
#     errors = _run_notebook(
#         'fairness_indicators/examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb')
#     self.assertEqual(errors, [])

#   def test_tfco_celeba(self):
#     errors = _run_notebook(
#         'fairness_indicators/examples/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb')
#     self.assertEqual(errors, [])

#   def test_tfco_wiki(self):
#     errors = _run_notebook(
#         'fairness_indicators/examples/Fairness_Indicators_TFCO_Wiki_Case_Study.ipynb')
#     self.assertEqual(errors, [])

#   def test_tfhub(self):
#     errors = _run_notebook(
#         'fairness_indicators/examples/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb')
#     self.assertEqual(errors, [])


if __name__ == '__main__':
  unittest.main()
