# What are the disadvantages of using classic decision tree algorithm for a large dataset?



### The computational efficiency perspective

It's a combinatorial search problem: at each split, we want to find the features that give us "the best bang for the buck" (maximizing information gain). If we choose a"brute" force approach, our computational complexity is O(m^2), where m is the number of features in our training set, and O(n^2) for the number of n training cases (I think it can be O(n log(n) if you are lucky).

Let's take a look at a simple dataset, Iris (150 flowers, 3 classes, 4 continuous features). At each split, we have to re-evaluate all 4 features, and for each feature we have to find the optimal value to split on, e.g,. sepal length <3.4 cm (this is for a binary split). Computational complexity is one of the reasons why people implement *binary* decision trees most of the time.

### The predictive performance perspective

An unpruned model is much more likely to overfit as a consequence of the curse of dimensionality. However, instead of pruning a single decision tree, it often a better idea to use ensemble methods. We could

- combine decision tree stumps that learn from each other by focusing on samples that are hard to classify (AdaBoost)
- create an ensemble of unpruned decision trees; draw bootstrap samples, and do random feature selection (random forests)
- forget about bagging and use all training samples as input for your unpruned trees; choose both the splitting feature and splitting value at random (= Extremely randomized trees)   

(Related topic: [How does the random forest model work? How is it different from bagging and boosting in ensemble models?](../bagging-boosting-rf.md))
