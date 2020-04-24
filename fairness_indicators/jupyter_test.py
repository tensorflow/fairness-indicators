import unittest
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


def get_errors(path):

  with open(path) as f:
    notebook = nbformat.read(f, as_version=4)

  proc = ExecutePreprocessor(timeout=None, kernel_name='python3')
  proc.allow_errors = True
  proc.preprocess(notebook, {'metadata': {'path': '/'}})

  errors = []
  for cell in notebook.cells:
    if 'outputs' in cell:
      for output in cell['outputs']:
        if output.output_type == 'error':
          errors.append(output)

  return errors


class TestNotebook(unittest.TestCase):

  def test_example(self):
    errors = get_errors(
        'examples/Fairness_Indicators_Example_Colab.ipynb')
    self.assertIsNone(errors, [])

  def test_facessd(self):
    errors = get_errors(
        'examples/Facessd_Fairness_Indicators_Example_Colab.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_lineage(self):
    errors = get_errors(
        'examples/Fairness_Indicators_Lineage_Case_Study.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tb(self):
    errors = get_errors(
        'examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tfco_celeba(self):
    errors = get_errors(
        'examples/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tfco_wiki(self):
    errors = get_errors(
        'examples/Fairness_Indicators_TFCO_Wiki_Case_Study.ipynb'
    )
    self.assertIsNone(errors, [])

  def test_tfhub(self):
    errors = get_errors(
        'examples/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb'
    )
    self.assertIsNone(errors, [])


if __name__ == '__main__':
  unittest.main()
