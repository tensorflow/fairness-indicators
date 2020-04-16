import unittest

import notebook_runner


class TestNotebook(unittest.TestCase):

  def test_runner(self):
    errors = notebook_runner.run_notebook(
        'fairness_indicators/examples/Fairness_Indicators_Example_Colab.ipynb')
    print('ERRORS (test)')
    print(type(errors))
    print(errors)
    self.assertEqual(errors, [])


if __name__ == '__main__':
  unittest.main()
