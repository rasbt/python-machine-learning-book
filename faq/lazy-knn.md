# Why is Nearest Neighbor a Lazy Algorithm?

Although, Nearest neighbor algorithms, for instance, the K-Nearest Neighbors (K-NN) for classification, are very "simple" algorithms, that's not why they are called *lazy* ;). K-NN is a lazy learner because it doesn't learn a discriminative function from the training data but "memorizes" the training dataset instead.

For example, the logistic regression algorithm learns its model weights (parameters) during training time. In contrast, there is no training time in K-NN. Although this may sound very convenient, this property doesn't come without a cost: The "prediction" step in K-NN is relatively expensive! Each time we want to make a prediction, K-NN is searching for the nearest neighbor(s) in the entire training set! (Note that there are certain tricks such as BallTrees and KDtrees to speed this up a bit.)

To summarize: An eager learner has a model fitting or training step. A lazy learner does not have a training phase.
