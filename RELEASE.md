<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug Fixes and Other Changes
*  Improve rendering of HTML stubs for Fairness Indicators UI

## Breaking Changes

## Deprecations

# Version 0.29.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow-data-validation>=0.29,<0.30`.
*   Depends on `tensorflow-model-analysis>=0.29,<0.30`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.28.0

## Major Features and Improvements

*   In Fairness Indicators UI, sort metrics list to show common metrics first
*   For lift, support negative values in bar chart.
*   Adding two new metrics - Flip Count and Flip Rate to evaluate Counterfactual
    Fairness.
*   Add Lift metrics under addons/fairness.
*   Porting Lift metrics into FI UI.

## Bug Fixes and Other Changes

*   Depends on `tensorflow-data-validation>=0.28,<0.29`.
*   Depends on `tensorflow-model-analysis>=0.28,<0.29`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.27.0

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Added test cases for DLVM testing.
*   Move the util files to a seperate folder.
*   Add `tensorflow-hub` as a dependency because it's used inside the
    example_model.py.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-data-validation>=0.27,<0.28`.
*   Depends on `tensorflow-model-analysis>=0.27,<0.28`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.26.0

## Major Features and Improvements

*   Sorting fairness metrics table rows to keep slices in order with slice drop
    down in the UI.

## Bug fixes and other changes

*   Update fairness_indicators.documentation.examples.util to TensorFlow 2.0.
*   Table now displays 3 decimal places instead of 2.
*   Fix the bug that metric list won't refresh if the input eval result changed.
*   Remove d3-tip dependency.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-data-validation>=0.26,<0.27`.
*   Depends on `tensorflow-model-analysis>=0.26,<0.27`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.25.0

## Major Features and Improvements

*   Add workflow buttons to Fairness Indicators UI, providing tutorial on how to
    configure metrics and parameters, and how to interpret the results.
*   Add metric definitions as tooltips in the metric selector UI
*   Removing prefix from metric names in graph titles in UI.
*   From this release Fairness Indicators will also be hosting nightly packages
    on https://pypi-nightly.tensorflow.org. To install the nightly package use
    the following command:

    ```
    pip install -i https://pypi-nightly.tensorflow.org/simple fairness-indicators
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of Fairness Indicators available on PyPI by
    running the command `pip install fairness-indicators` .

## Bug fixes and other changes

*   Update table colors.
*   Modify privacy note in Fairness Indicators UI.
*   Depends on `tensorflow-data-validation>=0.25,<0.26`.
*   Depends on `tensorflow-model-analysis>=0.25,<0.26`.

## Breaking changes

* N/A

## Deprecations

* N/A

# Version 0.24.0

## Major Features and Improvements

*   Made the Fairness Indicators UI thresholds drop down list sorted.

## Bug fixes and other changes

*   Fix in the issue where the Sort menu is not hidden when there is no model
    comparison.
*   Depends on `tensorflow-data-validation>=0.24,<0.25`.
*   Depends on `tensorflow-model-analysis>=0.24,<0.25`.

## Breaking changes

* N/A

## Deprecations

*   Deprecated Py3.5 support.

# Version 0.23.1

## Major Features and Improvements

* N/A

## Bug fixes and other changes

*  Fix broken import path in Fairness_Indicators_Example_Colab and Fairness_Indicators_on_TF_Hub_Text_Embeddings.

## Breaking changes

* N/A

## Deprecations

* N/A

# Version 0.23.0

## Major Features and Improvements

* N/A

## Bug fixes and other changes

*  Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
*  Depends on `tensorflow-data-validation>=0.23,<0.24`.
*  Depends on `tensorflow-model-analysis>=0.23,<0.24`.

## Breaking changes

* N/A

## Deprecations

*  Deprecating Py2 support.
*  Note: We plan to drop py3.5 support in the next release.
