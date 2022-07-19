# Fairness Indicators

![Fairness_Indicators](https://raw.githubusercontent.com/tensorflow/fairness-indicators/master/fairness_indicators/images/fairnessIndicators.png)

Fairness Indicators is designed to support teams in evaluating, improving, and comparing models for fairness concerns in partnership with the broader Tensorflow toolkit.

The tool is currently actively used internally by many of our products. We would love to partner with you to understand where Fairness Indicators is most useful, and where added functionality would be valuable. Please reach out at tfx@tensorflow.org. You can provide feedback and feature requests [here](https://github.com/tensorflow/fairness-indicators/issues/new/choose).

## Key links
* [Introductory Video](https://www.youtube.com/watch?v=pHT-ImFXPQo)
* [Fairness Indicators Case Study](https://developers.google.com/machine-learning/practica/fairness-indicators?utm_source=github&utm_medium=github&utm_campaign=fi-practicum&utm_term=&utm_content=repo-body)
* [Fairness Indicators Example Colab](https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Example_Colab.ipynb)
* [Pandas DataFrame to Fairness Indicators Case Study](https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Pandas_Case_Study.ipynb)
* [Fairness Indicators: Thinking about Fairness Evaluation](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/guide/guidance.md)

## What is Fairness Indicators?
Fairness Indicators enables easy computation of commonly-identified fairness metrics for **binary** and **multiclass** classifiers.

Many existing tools for evaluating fairness concerns don’t work well on large-scale datasets and models. At Google, it is important for us to have tools that can work on billion-user systems. Fairness Indicators will allow you to evaluate fairenss metrics across any size of use case.

In particular, Fairness Indicators includes the ability to:

* Evaluate the distribution of datasets
* Evaluate model performance, sliced across defined groups of users
  * Feel confident about your results with confidence intervals and evals at multiple thresholds
* Dive deep into individual slices to explore root causes and opportunities for improvement

This [case study](https://developers.google.com/machine-learning/practica/fairness-indicators?utm_source=github&utm_medium=github&utm_campaign=fi-practicum&utm_term=&utm_content=repo-body), complete with [videos](https://www.youtube.com/watch?v=pHT-ImFXPQo) and programming exercises, demonstrates how Fairness Indicators can be used on one of your own products to evaluate fairness concerns over time.

[![](http://img.youtube.com/vi/pHT-ImFXPQo/0.jpg)](http://www.youtube.com/watch?v=pHT-ImFXPQo "")

## [Installation](https://pypi.org/project/fairness-indicators/)

`pip install fairness-indicators`

The pip package includes:

* [**Tensorflow Data Validation (TFDV)**](https://github.com/tensorflow/data-validation) - analyze the distribution of your dataset
* [**Tensorflow Model Analysis (TFMA)**](https://github.com/tensorflow/model-analysis) - analyze model performance
  * **Fairness Indicators** - an addition to TFMA that adds fairness metrics and easy performance comparison across slices
* **The What-If Tool (WIT)**](https://github.com/PAIR-code/what-if-tool - an interactive visual interface designed to probe your models better

### Nightly Packages

Fairness Indicators also hosts nightly packages at
https://pypi-nightly.tensorflow.org on Google Cloud. To install the latest
nightly package, please use the following command:

```bash
pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple fairness-indicators
```

This will install the nightly packages for the major dependencies of Fairness
Indicators such as TensorFlow Data Validation (TFDV), TensorFlow Model Analysis
(TFMA).

## How can I use Fairness Indicators?
Tensorflow Models

* Access Fairness Indicators as part of the Evaluator component in Tensorflow Extended \[[docs](https://www.tensorflow.org/tfx/guide/evaluator)]
* Access Fairness Indicators in Tensorboard when evaluating other real-time metrics \[[docs](https://github.com/tensorflow/tensorboard/blob/master/docs/fairness-indicators.md)]

Not using existing Tensorflow tools? No worries!

* Download the Fairness Indicators pip package, and use Tensorflow Model Analysis as a standalone tool \[[docs](https://www.tensorflow.org/tfx/guide/fairness_indicators)]
* Model Agnostic TFMA enables you to compute Fairness Indicators based on the output of any model \[[docs](https://www.tensorflow.org/tfx/guide/fairness_indicators)]

## [Examples](https://github.com/tensorflow/fairness-indicators/tree/master/g3doc/tutorials) directory contains several examples.

* [Fairness_Indicators_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Example_Colab.ipynb) gives an overview of Fairness Indicators in [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma) and how to use it with a real dataset. This notebook also goes over [TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) and [What-If Tool](https://pair-code.github.io/what-if-tool/), two tools for analyzing TensorFlow models that are packaged with Fairness Indicators.
* [Fairness_Indicators_on_TF_Hub.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb) demonstrates how to use Fairness Indicators to compare models trained on different [text embeddings](https://en.wikipedia.org/wiki/Word_embedding). This notebook uses text embeddings from [TensorFlow Hub](https://www.tensorflow.org/hub), TensorFlow's library to publish, discover, and reuse model components.
* [Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb)
demonstrates how to visualize Fairness Indicators in TensorBoard.

## More questions?
For more information on how to think about fairness evaluation in the context of your use case, see [this link](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/guide/guidance.md).

If you have found a bug in Fairness Indicators, please file a [GitHub issue](https://github.com/tensorflow/fairness-indicators/issues/new/choose) with as much supporting information as you can provide.

## Compatible versions

The following table shows the  package versions that are
compatible with each other. This is determined by our testing framework, but
other *untested* combinations may also work.

|fairness-indicators                                                                        | tensorflow         | tensorflow-data-validation | tensorflow-model-analysis |
|-------------------------------------------------------------------------------------------|--------------------|----------------------------|---------------------------|
|[GitHub master](https://github.com/tensorflow/fairness-indicators/blob/master/RELEASE.md)  | nightly (1.x/2.x)  | 1.9.0                      | 0.40.0                    |
|[v0.40.0](https://github.com/tensorflow/fairness-indicators/blob/v0.40.0/RELEASE.md)       | 1.15.5 / 2.9       | 1.9.0                      | 0.40.0                    |
|[v0.39.0](https://github.com/tensorflow/fairness-indicators/blob/v0.39.0/RELEASE.md)       | 1.15.5 / 2.8       | 1.8.0                      | 0.39.0                    |
|[v0.38.0](https://github.com/tensorflow/fairness-indicators/blob/v0.38.0/RELEASE.md)       | 1.15.5 / 2.8       | 1.7.0                      | 0.38.0                    |
|[v0.37.0](https://github.com/tensorflow/fairness-indicators/blob/v0.37.0/RELEASE.md)       | 1.15.5 / 2.7       | 1.6.0                      | 0.37.0                    |
|[v0.36.0](https://github.com/tensorflow/fairness-indicators/blob/v0.36.0/RELEASE.md)       | 1.15.2 / 2.7       | 1.5.0                      | 0.36.0                    |
|[v0.35.0](https://github.com/tensorflow/fairness-indicators/blob/v0.35.0/RELEASE.md)       | 1.15.2 / 2.6       | 1.4.0                      | 0.35.0                    |
|[v0.34.0](https://github.com/tensorflow/fairness-indicators/blob/v0.34.0/RELEASE.md)       | 1.15.2 / 2.6       | 1.3.0                      | 0.34.0                    |
|[v0.33.0](https://github.com/tensorflow/fairness-indicators/blob/v0.33.0/RELEASE.md)       | 1.15.2 / 2.5       | 1.2.0                      | 0.33.0                    |
|[v0.30.0](https://github.com/tensorflow/fairness-indicators/blob/v0.30.0/RELEASE.md)       | 1.15.2 / 2.4       | 0.30.0                     | 0.30.0                    |
|[v0.29.0](https://github.com/tensorflow/fairness-indicators/blob/v0.29.0/RELEASE.md)       | 1.15.2 / 2.4       | 0.29.0                     | 0.29.0                    |
|[v0.28.0](https://github.com/tensorflow/fairness-indicators/blob/v0.28.0/RELEASE.md)       | 1.15.2 / 2.4       | 0.28.0                     | 0.28.0                    |
|[v0.27.0](https://github.com/tensorflow/fairness-indicators/blob/v0.27.0/RELEASE.md)       | 1.15.2 / 2.4       | 0.27.0                     | 0.27.0                    |
|[v0.26.0](https://github.com/tensorflow/fairness-indicators/blob/v0.26.0/RELEASE.md)       | 1.15.2 / 2.3       | 0.26.0                     | 0.26.0                    |
|[v0.25.0](https://github.com/tensorflow/fairness-indicators/blob/v0.25.0/RELEASE.md)       | 1.15.2 / 2.3       | 0.25.0                     | 0.25.0                    |
|[v0.24.0](https://github.com/tensorflow/fairness-indicators/blob/v0.24.0/RELEASE.md)       | 1.15.2 / 2.3       | 0.24.0                     | 0.24.0                    |
|[v0.23.0](https://github.com/tensorflow/fairness-indicators/blob/v0.23.0/RELEASE.md)       | 1.15.2 / 2.3       | 0.23.0                     | 0.23.0                    |
