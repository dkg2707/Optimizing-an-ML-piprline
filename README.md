# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.

This model is then compared to an Azure AutoML run.

## Summary
**This dataset contains data for a marketing campaign. The goal is to classify whether the person will subscribe to their services or not. It consists of data about the customer: age, marital status, how he/she was contacted, when they were contacted, do they have a loan, etc.**

**The best performing model was VotingEnsemble with an accuracy of 91.73%.**

## Scikit-learn Pipeline
    1. Data Collection.
    2. Data Cleaning.
    3. Test train split.
    4. Hyperparameter sampling: C - inverse regularization strength, max_iter - Max number of iteration for model to converge.
    5. Model train 
    6. Early stopping policy: Here we have used bandit policy which is based on slack factor calculated after 2 interval where training runs whose best metric is less than (best metric at interval 2/(1+slack_factor))
    7. Model Test
    8. Save model
    
    We have used hyperdrive to hypertune the parameters of logistic regression for this dataset.
    
    Also we have used AutoML to find the best model.


## AutoML
**AutoML generates the best model by running all the models on a dataset.**

**For this dataset the best model generated was VotingEnsemble with accuracy of 91.73%**

## Pipeline comparison
**Logistic Regression - Accuracy: 0.91259**

**Voting Ensemble - Accuracy:0.9173**

**AutoML had a variety of model it could evaluate from and hyperdrive only had logistic regression to evaluate with different paramerters. This the reason for AutoML's high accuracy.**

## Future work
**We could use different hyperdrive configurations, or we can use other model rather than logistic regression.**

**Explore the data mode to remove non-correlated features.**

**Different configurations for AutoML**
