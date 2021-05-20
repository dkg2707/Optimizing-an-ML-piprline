# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.

This model is then compared to an Azure AutoML run.

## Summary
**This dataset contains data for a marketing campaign. The goal is to classify whether the person will subscribe to their services or not. It consists of data about the customer: age, marital status, how he/she was contacted, when they were contacted, do they have a loan, etc.**

**The best performing model was VotingEnsemble with an accuracy of 91.73%.**

## Scikit-learn Pipeline
    1. Data Collection: The data is in the csv form It is being read from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv.
    2. Data Cleaning: Used one hot encoding for contact, job, education. Dropped rows with null values. Converted columns like poutcome, housing, default, loan and marital to binary input i.e. 1 or 0. 
    3. Test train split: To test the accuracy of the trained model we split data into train and test. Accuracy of trained model is checked on test data. 
    4. Hyperparameter sampling: C - inverse regularization strength, max_iter - Max number of iteration for model to converge.
    5. Model train: Model is trained
    6. Early stopping policy: Here we have used bandit policy which is based on slack factor calculated after 2 interval where training runs whose best metric is less than (best metric at interval 2/(1+slack_factor))
    7. Model Test: Model is tested and accuracy is calculated to evaluate the model.
    8. Save model: Best model is saved for further use.
    
We have used hyperdrive to hypertune the parameters of logistic regression for this dataset.
    
RandomParameter sampling is chosen for hyperdrive because it chooses random random combination of parameters which sometimes may be able to find the best model quickly and it is great for discovery learning.
    
We have used bandit policy for early stopping because we want to stop the exexution of training the model if stopping policy is met. It will reduce the overall computation cost.

Also we have used AutoML to find the best model.


## AutoML
**AutoML generates the best model by running all the models on a dataset. It can perform regression, classification and time series forecasting. We provide a time-out interval for automl to help save computational cost. It also automatically does the data preparation part.**

**In our case AutoML Guardrails alerted us to balance the classes as the model may be biased to one class.**

Best model chosen by AutoML:

Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=None, enable_feature_sweeping=None,
                                 feature_sweeping_config=None,
                                 feature_sweeping_timeout=None,
                                 featurization_config=None, force_text_dnn=None,
                                 is_cross_validation=None,
                                 is_onnx_compatible=None, logger=None,
                                 observer=None, task=None, working_dir=None)),
                ('prefittedsoftvotingclassifier',...
                                                                                               max_leaves=15,
                                                                                               min_child_weight=1,
                                                                                               missing=nan,
                                                                                               n_estimators=25,
                                                                                               n_jobs=1,
                                                                                               nthread=None,
                                                                                               objective='reg:logistic',
                                                                                               random_state=0,
                                                                                               reg_alpha=0,
                                                                                               reg_lambda=0.5208333333333334,
                                                                                               scale_pos_weight=1,
                                                                                               seed=None,
                                                                                               silent=None,
                                                                                               subsample=0.6,
                                                                                               tree_method='auto',
                                                                                               verbose=-10,
                                                                                               verbosity=0))],
                                                                     verbose=False))],
                                               flatten_transform=None,
                                               weights=[0.2, 0.1, 0.3, 0.1, 0.1,
                                                        0.1, 0.1]))],
         verbose=False)

**For this dataset the best model generated was VotingEnsemble with accuracy of 91.73%**

## Pipeline comparison
**Logistic Regression - Accuracy: 0.91259**

**Voting Ensemble - Accuracy:0.9173**

**AutoML had a variety of model it could evaluate from and hyperdrive only had logistic regression to evaluate with different paramerters. This the reason for AutoML's high accuracy.**

## Future work
**We could use different hyperdrive configurations: We could have used a bigger slack factor to give more time to converge the model but thhis would have been computationally expensive. Other option would have been to choose another sampling method like Bayesian Parameter Sampling as it samples from a normal distribution which could have resulted in better accuracy.**

**For AutoML we could have increased the stopping time to give more time to AutoML to find better models.**

**Explore the data more to remove non-correlated features which can reduce the computational costs and give better accuracy**
