<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.6.0,<3`.

## Breaking Changes

## Deprecations

# Version 0.33.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorboard>=2.5.0,<3`.
*   Depends on `tensorflow>=2.5.0,<3`.
*   Depends on `protobuf>=3.13,<4`.
*   Depends on `tensorflow-model-analysis>=0.33,<0.34`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.30.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorboard>=2.4.0,!=2.5.*,<3`.
*   Depends on `tensorflow>=2.4.0,!=2.5.*,<3`.
*   Depends on `tensorflow-model-analysis>=0.30,<0.31`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.29.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow-model-analysis>=0.29,<0.30`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.28.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow-model-analysis>=0.28,<0.29`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.27.0

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Depends on `tensorboard>=2.4.0,<3`.
*   Depends on `tensorflow>=2.4.0,<3`.
*   Depends on `tensorflow-model-analysis>=0.27,<0.28`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.26.0

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Depends on `tensorboard>=2.3.0,!=2.4.*,<3`.
*   Depends on `tensorflow>=2.3.0,!=2.4.*,<3`.
*   Depends on `tensorflow-model-analysis>=0.26,<0.27`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

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

# Version 0.24.0

## Major Features and Improvements

* N/A

## Bug fixes and other changes

*   Fix in the error message while rendering evaluation results in
    TensorBoard plugin from evaluation output path provided in the URL.
*   Depends on `tensorflow-model-analysis>=0.24,<0.25`.

## Breaking changes

* N/A

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
