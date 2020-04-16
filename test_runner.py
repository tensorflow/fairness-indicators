import unittest

import notebook_runner


class TestNotebook(unittest.TestCase):

  def test_example(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_Example_Colab.ipynb')
    self.assertEmpty(errors)

  def test_facessd(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Facessd_Fairness_Indicators_Example_Colab.ipynb')
    self.assertEmpty(errors)

  def test_lineage(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_Lineage_Case_Study.ipynb')
    self.assertEmpty(errors)

  def test_tb(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb')
    self.assertEmpty(errors)

  def test_tfco_celeba(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb')
    self.assertEmpty(errors)

  def test_tfco_wiki(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_TFCO_Wiki_Case_Study.ipynb')
    self.assertEmpty(errors)

  def test_tfhub(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb')
    self.assertEmpty(errors)


if __name__ == '__main__':
  unittest.main()
