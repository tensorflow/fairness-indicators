# Fairness Indicators BETA 
Fairness Indicators is designed to support teams in evaluating and improving models for fairness concerns in partnership with the broader Tensorflow toolkit.  

The tool is currently actively used internally by many of our products, and is now available in BETA for you to try for your own use cases. 

We would love to partner with you to understand where Fairness Indicators is most useful, and where added functionality would be valuable. Please reach out at ml-fairness-partnerships@google.com. You can provide any feedback on your experience, and feature requests, here. 

## What is Fairness Indicators? 
Fairness Indicators enables easy computation of commonly-identified fairness metrics for **binary** and **multiclass** classifiers. 

Many existing tools for evaluating fairness concerns don’t work well on large scale datasets and models. At Google, it is important for us to have tools that can work on billion-user systems. Fairness Indicators will allow you to evaluate across any size of use case. 

In particular, Fairness Indicators includes the ability to:
* Evaluate the distribution of datasets
* Evaluate model performance, sliced across defined groups of users 
  * Feel confident about your results with confidence intervals and evals at multiple thresholds 
  * Ensure user privacy with K-anonymity configuration
* Dive deep into individual slices to explore root causes and opportunities for improvement

This Introductory Video provides an example of how Fairness Indicators can be used on of our own products to evaluate fairness concerns overtime. This Demo Colab provides a hands-on experience of using Fairness Indicators.

The pip package download includes: 
* **Tensorflow Data Analysis (TFDV)** \[analyze distribution of your dataset] 
* **Tensorflow Model Analysis (TFMA)** \[analyze model performance] 
  * **Fairness Indicators** \[an addition to TFMA that adds fairness metrics and the ability to easily compare performance across slices]  
* **The What-If Tool (WIT)** \[an interactive visual interface designed to probe your models better]

## How can I use Fairness Indicators?
Tensorflow Models
* Access Fairness Indicators as part of the Evaluator component in Tensorflow Extended \[documentation for Evaluator, example colab]
* Access Fairness Indicators in Tensorboard when evaluating other real-time metrics \[documentation for Fairness Indicators for Tensorboard]

Not using existing Tensorflow tools? No worries!
* Download the Fairness Indicators pip package, and use Tensorflow Model Analysis as a standalone tool [documentation for TFMA] 

Non-Tensorflow Models
* Model Agnostic TFMA enables you to compute Fairness Indicators based on the output of any model \[documentation for Model Agnostic TFMA]

## Example Colab

## Guidance and Metrics
