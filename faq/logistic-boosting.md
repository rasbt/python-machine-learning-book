# Do bagging and boosting can be used with logistic regression?

I am not sure if bagging would make much sense for logistic regression -- in bagging, we reduce the variance of the deep decision tree models that overfit the training data, which wouldn't really apply to logistic regression.
 
Boosting could work though, however, I think that "stacking" would be a better approach here. Stacking would be more "powerful" since we don't use a pre-specified equation to adjust the weight, rather, we train a meta-classifier to learn the optimal weights to combine the models.
  
  
Here's one of the many interesting, related papers, I recommend you to check out :)

- "Is Combining Classifiers with Stacking Better than Selecting the Best One?" 
