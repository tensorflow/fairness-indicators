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
