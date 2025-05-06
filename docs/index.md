# Fairness Indicators

/// html | div[style='float: left; width: 50%;']
Fairness Indicators is a library that enables easy computation of commonly-identified fairness metrics for binary and multiclass classifiers. With the Fairness Indicators tool suite, you can:

- Compute commonly-identified fairness metrics for classification models
- Compare model performance across subgroups to a baseline, or to other models
- Use confidence intervals to surface statistically significant disparities
- Perform evaluation over multiple thresholds

Use Fairness Indicators via the:

- [Evaluator component](https://tensorflow.github.io/tfx/guide/evaluator/) in a [TFX pipeline](https://tensorflow.github.io/tfx/)
- [TensorBoard plugin](https://github.com/tensorflow/tensorboard/blob/master/docs/fairness-indicators.md)
- [TensorFlow Model Analysis library](https://www.tensorflow.org/tfx/guide/fairness_indicators)
- [Model Agnostic TFMA library](https://www.tensorflow.org/tfx/guide/fairness_indicators#using_fairness_indicators_with_non-tensorflow_models)
<!-- TODO: Change the TFMA link when the new docs are deployed -->
///

/// html | div[style='float: right;width: 50%;']
```python
eval_config_pbtxt = """

model_specs {
    label_key: "%s"
}

metrics_specs {
    metrics {
        class_name: "FairnessIndicators"
        config: '{ "thresholds": [0.25, 0.5, 0.75] }'
    }
    metrics {
        class_name: "ExampleCount"
    }
}

slicing_specs {}
slicing_specs {
    feature_keys: "%s"
}

options {
    compute_confidence_intervals { value: False }
    disabled_outputs{values: "analysis"}
}
""" % (LABEL_KEY, GROUP_KEY)
```
///

/// html | div[style='clear: both;']
///

<div class="grid cards" markdown>

-   ![asdf](https://www.tensorflow.org/static/responsible_ai/fairness_indicators/images/mlpracticum_480.png)

    ### [ML Practicum: Fairness in Perspective API using Fairness Indicators](https://developers.google.com/machine-learning/practica/fairness-indicators?utm_source=github&utm_medium=github&utm_campaign=fi-practicum&utm_term=&utm_content=repo-body)

    ---

    [Try the Case Study](https://developers.google.com/machine-learning/practica/fairness-indicators?utm_source=github&utm_medium=github&utm_campaign=fi-practicum&utm_term=&utm_content=repo-body)

-   ![Fairness Indicators on the TensorFlow blog](../images/tf_full_color_primary_icon.svg)

    ### [Fairness Indicators on the TensorFlow blog](https://blog.tensorflow.org/2019/12/fairness-indicators-fair-ML-systems.html)
    
    ---

    [Read on the TensorFlow blog](https://blog.tensorflow.org/2019/12/fairness-indicators-fair-ML-systems.html)

-   ![Fairness Indicators on GitHub](https://www.tensorflow.org/static/resources/images/github-card-16x9_480.png)

    ### [Fairness Indicators on GitHub](https://github.com/tensorflow/fairness-indicators)
    ---

    [View on GitHub](https://github.com/tensorflow/fairness-indicators)

-   ![Fairness Indicators on the Google AI Blog](https://www.tensorflow.org/static/responsible_ai/fairness_indicators/images/googleai_720.png)

    ### [Fairness Indicators on the Google AI Blog](https://ai.googleblog.com/2019/12/fairness-indicators-scalable.html)
    ---

    [Read on Google AI blog](https://ai.googleblog.com/2019/12/fairness-indicators-scalable.html)

-   ![type:video](https://www.youtube.com/watch?v=6CwzDoE8J4M)

    ### [Fairness Indicators at Google I/O](https://www.youtube.com/watch?v=6CwzDoE8J4M)

    ---

    [Watch the video](https://www.youtube.com/watch?v=6CwzDoE8J4M)

</div>
