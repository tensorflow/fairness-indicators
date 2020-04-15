import unittest

import notebook_runner


class TestNotebook(unittest.TestCase):

  def test_runner(self):
    errors = notebook_runner.run_notebook(
        'examples/Fairness_Indicators_Example_Colab.ipynb')
    self.assertEqual(errors, [])


if __name__ == '__main__':
  unittest.main()
