# Evaluating Models with the Fairness Indicators Dashboard [Beta]

![Fairness Indicators](https://raw.githubusercontent.com/tensorflow/tensorboard/master/docs/images/fairness-indicators.png)

Fairness Indicators for TensorBoard enables easy computation of
commonly-identified fairness metrics for _binary_ and _multiclass_ classifiers.
With the plugin, you can visualize fairness evaluations for your runs and easily
compare performance across groups.

In particular, Fairness Indicators for TensorBoard allows you to evaluate and
visualize model performance, sliced across defined groups of users. Feel
confident about your results with confidence intervals and evaluations at
multiple thresholds.

Many existing tools for evaluating fairness concerns don’t work well on large
scale datasets and models. At Google, it is important for us to have tools that
can work on billion-user systems. Fairness Indicators will allow you to evaluate
across any size of use case, in the TensorBoard environment or in
[Colab](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/).

## Requirements

To install Fairness Indicators for TensorBoard, run:

```
python3 -m virtualenv ~/tensorboard_demo
source ~/tensorboard_demo/bin/activate
pip install --upgrade pip
pip install fairness_indicators
pip install tensorboard-plugin-fairness-indicators
```

## Demo Colab

[Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb)
contains an end-to-end demo to train and evaluate a model and visualize fairness evaluation
results in TensorBoard.

## Usage

To use the Fairness Indicators with your own data and evaluations:

1.  Train a new model and evaluate using
    `tensorflow_model_analysis.run_model_analysis` or
    `tensorflow_model_analysis.ExtractEvaluateAndWriteResult` API in
    [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py).
    For code snippets on how to do this, see the Fairness Indicators colab
    [here](https://github.com/tensorflow/fairness-indicators).

2.  Write a summary data file using [`demo.py`](https://github.com/tensorflow/fairness-indicators/blob/master/tensorboard_plugin/tensorboard_plugin_fairness_indicators/demo.py), which will be read
    by TensorBoard to render the Fairness Indicators dashboard (See the
    [TensorBoard tutorial](https://github.com/tensorflow/tensorboard/blob/master/README.md)
    for more information on summary data files).

    Flags to be used with the `demo.py` utility:

    -   `--logdir`: Directory where TensorBoard will write the summary
    -   `--eval_result_output_dir`: Directory containing evaluation results
        evaluated by TFMA

    ```
    python demo.py --logdir=<logdir> --eval_result_output_dir=<eval_result_dir>`
    ```

    Or you can also use `tensorboard_plugin_fairness_indicators.summary_v2` API to write the summary file.

    ```
    writer = tf.summary.create_file_writer(<logdir>)
    with writer.as_default():
        summary_v2.FairnessIndicators(<eval_result_dir>, step=1)
    writer.close()
    ```

3.  Run TensorBoard

    Note: This will start a local instance. After the local instance is started, a link
    will be displayed to the terminal. Open the link in your browser to view the
    Fairness Indicators dashboard.

    -   `tensorboard --logdir=<logdir>`
    -   Select the new evaluation run using the drop-down on the left side of
        the dashboard to visualize results.
