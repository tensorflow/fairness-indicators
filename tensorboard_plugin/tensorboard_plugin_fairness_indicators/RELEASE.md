# Version 0.25.0

## Major Features and Improvements

*   From this release Tensorboard Plugin will also be hosting nightly packages
    on https://pypi-nightly.tensorflow.org. To install the nightly package use
    the following command:

    ```
    pip install -i https://pypi-nightly.tensorflow.org/simple tensorboard-plugin-fairness-indicators
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of Tensorboard Plugin available on PyPI by
    running the command `pip install tensorboard-plugin-fairness-indicators` .

## Bug fixes and other changes

*   Adding support for model comparison using dynamic URL in TensorBoard plugin.
*   Depends on `tensorflow-model-analysis>=0.25,<0.26`.

## Breaking changes

* N/A

## Deprecations

* N/A
