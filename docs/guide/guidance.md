# Fairness Indicators: Thinking about Fairness Evaluation

Fairness Indicators is a useful tool for evaluating _binary_ and _multi-class_
classifiers for fairness. Eventually, we hope to expand this tool, in
partnership with all of you, to evaluate even more considerations.

Keep in mind that quantitative evaluation is only one part of evaluating a
broader user experience. Start by thinking about the different _contexts_
through which a user may experience your product. Who are the different types of
users your product is expected to serve? Who else may be affected by the
experience?

When considering AI's impact on people, it is important to always remember that
human societies are extremely complex! Understanding people, and their social
identities, social structures and cultural systems are each huge fields of open
research in their own right. Throw in the complexities of cross-cultural
differences around the globe, and getting even a foothold on understanding
societal impact can be challenging. Whenever possible, it is recommended you
consult with appropriate domain experts, which may include social scientists,
sociolinguists, and cultural anthropologists, as well as with members of the
populations on which technology will be deployed.

A single model, for example, the toxicity model that we leverage in the
[example colab](https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_Example_Colab),
can be used in many different contexts. A toxicity model deployed on a website
to filter offensive comments, for example, is a very different use case than the
model being deployed in an example web UI where users can type in a sentence and
see what score the model gives. Depending on the use case, and how users
experience the model prediction, your product will have different risks,
effects, and opportunities and you may want to evaluate for different fairness
concerns.

The questions above are the foundation of what ethical considerations, including
fairness, you may want to take into account when designing and developing your
ML-based product. These questions also motivate which metrics and which groups
of users you should use the tool to evaluate.

Before diving in further, here are three recommended resources for getting
started:

*   **[The People + AI Guidebook](https://pair.withgoogle.com/) for
    Human-centered AI design:** This guidebook is a great resource for the
    questions and aspects to keep in mind when designing a machine-learning
    based product. While we created this guidebook with designers in mind, many
    of the principles will help answer questions like the one posed above.
*   **[Our Fairness Lessons Learned](https://www.youtube.com/watch?v=6CwzDoE8J4M):**
    This talk at Google I/O discusses lessons we have learned in our goal to
    build and design inclusive products.
*   **[ML Crash Course: Fairness](https://developers.google.com/machine-learning/crash-course/fairness/video-lecture):**
    The ML Crash Course has a 70 minute section dedicated to identifying and
    evaluating fairness concerns

So, why look at individual slices? Evaluation over individual slices is
important as strong overall metrics can obscure poor performance for certain
groups. Similarly, performing well for a certain metric (accuracy, AUC) doesn’t
always translate to acceptable performance for other metrics (false positive
rate, false negative rate) that are equally important in assessing opportunity
and harm for users.

The below sections will walk through some of the aspects to consider.

## Which groups should I slice by?

In general, a good practice is to slice by as many groups as may be affected by
your product, since you never know when performance might differ for one of the
other. However, if you aren’t sure, think about the different users who may be
engaging with your product, and how they might be affected. Consider,
especially, slices related to sensitive characteristics such as race, ethnicity,
gender, nationality, income, sexual orientation, and disability status.

**What if I don’t have data labeled for the slices I want to investigate?**

Good question. We know that many datasets don’t have ground-truth labels for
individual identity attributes.

If you find yourself in this position, we recommend a few approaches:

1.  Identify if there _are_ attributes that you have that may give you some
    insight into the performance across groups. For example, _geography_ while
    not equivalent to ethnicity & race, may help you uncover any disparate
    patterns in performance
1.  Identify if there are representative public datasets that might map well to
    your problem. You can find a range of diverse and inclusive datasets on the
    [Google AI site](https://ai.google/responsibilities/responsible-ai-practices/?category=fairness),
    which include
    [Project Respect](https://www.blog.google/technology/ai/fairness-matters-promoting-pride-and-respect-ai/),
    [Inclusive Images](https://www.kaggle.com/c/inclusive-images-challenge), and
    [Open Images Extended](https://ai.google/tools/datasets/open-images-extended-crowdsourced/),
    among others.
1.  Leverage rules or classifiers, when relevant, to label your data with
    objective surface-level attributes. For example, you can label text as to
    whether or not there is an identity term _in_ the sentence. Keep in mind
    that classifiers have their own challenges, and if you’re not careful, may
    introduce another layer of bias as well. Be clear about what your classifier
    is <span style="text-decoration:underline;">actually</span> classifying. For
    example, an age classifier on images is in fact classifying _perceived age_.
    Additionally, when possible, leverage surface-level attributes that _can_ be
    objectively identified in the data. For example, it is ill-advised to build
    an image classifier for race or ethnicity, because these are not visual
    traits that can be defined in an image. A classifier would likely pick up on
    proxies or stereotypes. Instead, building a classifier for skin tone may be
    a more appropriate way to label and evaluate an image. Lastly, ensure high
    accuracy for classifiers labeling such attributes.
1.  Find more representative data that is labeled

**Always make sure to evaluate on multiple, diverse datasets.**

If your evaluation data is not adequately representative of your user base, or
the types of data likely to be encountered, you may end up with deceptively good
fairness metrics. Similarly, high model performance on one dataset doesn’t
guarantee high performance on others.

**Keep in mind subgroups aren’t always the best way to classify individuals.**

People are multidimensional and belong to more than one group, even within a
single dimension -- consider someone who is multiracial, or belongs to multiple
racial groups. Also, while overall metrics for a given racial group may look
equitable, particular interactions, such as race and gender together may show
unintended bias. Moreover, many subgroups have fuzzy boundaries which are
constantly being redrawn.

**When have I tested enough slices, and how do I know which slices to test?**

We acknowledge that there are a vast number of groups or slices that may be
relevant to test, and when possible, we recommend slicing and evaluating a
diverse and wide range of slices and then deep-diving where you spot
opportunities for improvement. It is also important to acknowledge that even
though you may not see concerns on slices you have tested, that doesn’t imply
that your product works for _all_ users, and getting diverse user feedback and
testing is important to ensure that you are continually identifying new
opportunities.

To get started, we recommend thinking through your particular use case and the
different ways users may engage with your product. How might different users
have different experiences? What does that mean for slices you should evaluate?
Collecting feedback from diverse users may also highlight potential slices to
prioritize.

## Which metrics should I choose?

When selecting which metrics to evaluate for your system, consider who will be
experiencing your model, how it will be experienced, and the effects of that
experience.

For example, how does your model give people more dignity or autonomy, or
positively impact their emotional, physical or financial wellbeing? In contrast,
how could your model’s predictions reduce people's dignity or autonomy, or
negatively impact their emotional, physical or financial wellbeing?

**In general, we recommend slicing _all your existing performance metrics as
good practice. We also recommend evaluating your metrics across
<span style="text-decoration:underline;">multiple thresholds</span>_** in order
to understand how the threshold can affect the performance for different groups.

In addition, if there is a predicted label which is uniformly "good" or “bad”,
then consider reporting (for each subgroup) the rate at which that label is
predicted. For example, a “good” label would be a label whose prediction grants
a person access to some resource, or enables them to perform some action.

## Critical fairness metrics for classification

When thinking about a classification model, think about the effects of _errors_
(the differences between the actual “ground truth” label, and the label from the
model). If some errors may pose more opportunity or harm to your users, make
sure you evaluate the rates of these errors across groups of users. These error
rates are defined below, in the metrics currently supported by the Fairness
Indicators beta.

**Over the course of the next year, we hope to release case studies of different
use cases and the metrics associated with these so that we can better highlight
when different metrics might be most appropriate.**

**Metrics available today in Fairness Indicators**

Note: There are many valuable fairness metrics that are not currently supported
in the Fairness Indicators beta. As we continue to add more metrics, we will
continue to add guidance for these metrics, here. Below, you can access
instructions to add your own metrics to Fairness Indicators. Additionally,
please reach out to [tfx@tensorflow.org](mailto:tfx@tensorflow.org) if there are
metrics that you would like to see. We hope to partner with you to build this
out further.

**Positive Rate / Negative Rate**

*   _<span style="text-decoration:underline;">Definition:</span>_ The percentage
    of data points that are classified as positive or negative, independent of
    ground truth
*   _<span style="text-decoration:underline;">Relates to:</span>_ Demographic
    Parity and Equality of Outcomes, when equal across subgroups
*   _<span style="text-decoration:underline;">When to use this metric:</span>_
    Fairness use cases where having equal final percentages of groups is
    important

**True Positive Rate / False Negative Rate**

*   _<span style="text-decoration:underline;">Definition:</span>_ The percentage
    of positive data points (as labeled in the ground truth) that are
    _correctly_ classified as positive, or the percentage of positive data
    points that are _incorrectly_ classified as negative
*   _<span style="text-decoration:underline;">Relates to:</span>_ Equality of
    Opportunity (for the positive class), when equal across subgroups
*   _<span style="text-decoration:underline;">When to use this metric:</span>_
    Fairness use cases where it is important that the same % of qualified
    candidates are rated positive in each group. This is most commonly
    recommended in cases of classifying positive outcomes, such as loan
    applications, school admissions, or whether content is kid-friendly

**True Negative Rate / False Positive Rate**

*   _<span style="text-decoration:underline;">Definition:</span>_ The percentage
    of negative data points (as labeled in the ground truth) that are correctly
    classified as negative, or the percentage of negative data points that are
    incorrectly classified as positive
*   _<span style="text-decoration:underline;">Relates to:</span>_ Equality of
    Opportunity (for the negative class), when equal across subgroups
*   _<span style="text-decoration:underline;">When to use this metric:</span>_
    Fairness use cases where error rates (or misclassifying something as
    positive) are more concerning than classifying the positives. This is most
    common in abuse cases, where _positives_ often lead to negative actions.
    These are also important for Facial Analysis Technologies such as face
    detection or face attributes

Note: When both “positive” and “negative” mistakes are equally important, the
metric is called “equality of
<span style="text-decoration:underline;">odds</span>”. This can be measured by
evaluating and aiming for equality across both the TNR & FNR, or both the TPR &
FPR. For example, an app that counts how many cars go past a stop sign is
roughly equally bad whether or not it accidentally includes an extra car (a
false positive) or accidentally excludes a car (a false negative).

**Accuracy & AUC**

*   _<span style="text-decoration:underline;">Relates to:</span>_ Predictive
    Parity, when equal across subgroups
*   _<span style="text-decoration:underline;">When to use these metrics:</span>_
    Cases where precision of the task is most critical (not necessarily in a
    given direction), such as face identification or face clustering

**False Discovery Rate**

*   _<span style="text-decoration:underline;">Definition:</span>_ The percentage
    of negative data points (as labeled in the ground truth) that are
    incorrectly classified as positive out of all data points classified as
    positive. This is also the inverse of PPV
*   _<span style="text-decoration:underline;">Relates to:</span>_ Predictive
    Parity (also known as Calibration), when equal across subgroups
*   _<span style="text-decoration:underline;">When to use this metric:</span>_
    Cases where the fraction of correct positive predictions should be equal
    across subgroups

**False Omission Rate**

*   _<span style="text-decoration:underline;">Definition:</span>_ The percentage
    of positive data points (as labeled in the ground truth) that are
    incorrectly classified as negative out of all data points classified as
    negative. This is also the inverse of NPV
*   _<span style="text-decoration:underline;">Relates to:</span>_ Predictive
    Parity (also known as Calibration), when equal across subgroups
*   _<span style="text-decoration:underline;">When to use this metric:</span>_
    Cases where the fraction of correct negative predictions should be equal
    across subgroups

Note: When used together, False Discovery Rate and False Omission Rate relate to
Conditional Use Accuracy Equality, when FDR and FOR are both equal across
subgroups. FDR and FOR are also similar to FPR and FNR, where FDR/FOR compare
FP/FN to predicted negative/positive data points, and FPR/FNR compare FP/FN to
ground truth negative/positive data points. FDR/FOR can be used instead of
FPR/FNR when predictive parity is more critical than equality of opportunity.

**Overall Flip Rate / Positive to Negative Prediction Flip Rate / Negative to
Positive Prediction Flip Rate**

*   *<span style="text-decoration:underline;">Definition:</span>* The
    probability that the classifier gives a different prediction if the identity
    attribute in a given feature were changed.
*   *<span style="text-decoration:underline;">Relates to:</span>* Counterfactual
    fairness
*   *<span style="text-decoration:underline;">When to use this metric:</span>*
    When determining whether the model’s prediction changes when the sensitive
    attributes referenced in the example is removed or replaced. If it does,
    consider using the Counterfactual Logit Pairing technique within the
    Tensorflow Model Remediation library.

**Flip Count / Positive to Negative Prediction Flip Count / Negative to Positive
Prediction Flip Count** *

*   *<span style="text-decoration:underline;">Definition:</span>* The number of
    times the classifier gives a different prediction if the identity term in a
    given example were changed.
*   *<span style="text-decoration:underline;">Relates to:</span>* Counterfactual
    fairness
*   *<span style="text-decoration:underline;">When to use this metric:</span>*
    When determining whether the model’s prediction changes when the sensitive
    attributes referenced in the example is removed or replaced. If it does,
    consider using the Counterfactual Logit Pairing technique within the
    Tensorflow Model Remediation library.

**Examples of which metrics to select**

*   _Systematically failing to detect faces in a camera app can lead to a
    negative user experience for certain user groups._ In this case, false
    negatives in a face detection system may lead to product failure, while a
    false positive (detecting a face when there isn’t one) may pose a slight
    annoyance to the user. Thus, evaluating and minimizing the false negative
    rate is important for this use case.
*   _Unfairly marking text comments from certain people as “spam” or “high
    toxicity” in a moderation system leads to certain voices being silenced._ On
    one hand, a high false positive rate leads to unfair censorship. On the
    other, a high false negative rate could lead to a proliferation of toxic
    content from certain groups, which may both harm the user and constitute a
    representational harm for those groups. Thus, both metrics are important to
    consider, in addition to metrics which take into account all types of errors
    such as accuracy or AUC.

**Don’t see the metrics you’re looking for?**

Follow the documentation
[here](https://github.com/tensorflow/model-analysis/blob/master/g3doc/post_export_metrics.md)
to add you own custom metric.

## Final notes

**A gap in metric between two groups can be a sign that your model may have
unfair skews**. You should interpret your results according to your use case.
However, the first sign that you may be treating one set of users _unfairly_ is
when the metrics between that set of users and your overall are significantly
different. Make sure to account for confidence intervals when looking at these
differences. When you have too few samples in a particular slice, the difference
between metrics may not be accurate.

**Achieving equality across groups on Fairness Indicators doesn’t mean the model
is fair.** Systems are highly complex, and achieving equality on one (or even
all) of the provided metrics can’t guarantee Fairness.

**Fairness evaluations should be run throughout the development process and
post-launch (not the day before launch).** Just like improving your product is
an ongoing process and subject to adjustment based on user and market feedback,
making your product fair and equitable requires ongoing attention. As different
aspects of the model changes, such as training data, inputs from other models,
or the design itself, fairness metrics are likely to change. “Clearing the bar”
once isn’t enough to ensure that all of the interacting components have remained
intact over time.

**Adversarial testing should be performed for rare, malicious examples.**
Fairness evaluations aren’t meant to replace adversarial testing. Additional
defense against rare, targeted examples is crucial as these examples probably
will not manifest in training or evaluation data.
