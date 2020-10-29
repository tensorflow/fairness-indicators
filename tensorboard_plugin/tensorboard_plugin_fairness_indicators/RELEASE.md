<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version(Still in Development)

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

## Breaking changes

## Deprecations

# Version 0.24.0

## Major Features and Improvements

* N/A

## Bug fixes and other changes

*   Fix in the error message while rendering evaluation results in
    TensorBoard plugin from evaluation output path provided in the URL.
*   Adding support for model comparison using dynamic URL in TensorBoard plugin.
*   Depends on `tensorflow-model-analysis>=0.24,<0.25`.

## Breaking changes

*   Depends on `tensorflow-model-analysis>=0.23,<0.24`.

## Deprecations

*   Deprecating Py3.5 support.

# Version 0.23.0

## Major Features and Improvements

* N/A

## Bug fixes and other changes

*   Depends on `tensorboard>=2.3.0,<3`.
*   Depends on `tensorflow>=2.3.0,<3`.
*   Depends on `tensorflow-model-analysis>=0.23,<0.24`.
*   Adding model comparison support in TensorBoard Plugin.

## Breaking changes

* N/A

## Deprecations

*   Deprecating Py2 support.
*   Note: We plan to drop py3.5 support in the next release.
