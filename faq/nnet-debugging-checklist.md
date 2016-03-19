# How do I debug an artificial neural network algorithm?

There are many, many reasons that can explain a unexpected, "bad" performance of neural networks. Let's compile a quick check list that we can process in a somewhat sequential manner to get to the root of that problem

1. Is the data set okay? More concretely: Is there a lot of noise? Are the features "powerful" enough to discriminate between classes? (It's a good idea to try a bunch of off-the-shelf classifiers to get an initial benchmark; classifiers like random forest, softmax regression, or kernel SVM)
2. Did we forget standardizing the features?
3. Did we implement and use gradient checking to make sure that our implementation is correct?
4. Do we use a random weight initialization scheme (e.g., from a random normal distribution multiplied by a small coefficient < 0) vs. initializing the model parameters to all-zero weights?
5. Did we try to increase or decrease the learning rate?
6. Have we checked that the cost decreases over time? If yes, have we tried to increase the number of epochs?
7. Have we tried to modify the learning rate using momentum learning and/or a decrease constant (e.g., AdaGrad)
8. Have we tried different non-linear activation functions other than the one we are currently using (e.g., logistic sigmoid, tanh, or ReLU)?
9. When you estimated the performance of our network via cross-validation (for example, via holdout or k-fold), did we notice a large discrepancy between training and validation performance? A substantial difference in performance on training and validation sets may indicate that we are overfitting the training data too much. As a remedy, we could try to
    1. collect more training samples if possible
    2. decrease the complexity of your network (e.g,. fewer nodes, fewer hidden layers)
    3. implement dropout
    4. add a penalty against complexity to the cost function (e.g., L2 regularization)
