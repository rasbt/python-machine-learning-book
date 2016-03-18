# How do I debug an artificial neural network algorithm?

There are many, many reasons that can explain a unexpected, "bad" performance of neural networks. Let's compile a quick check list that we can process in a somewhat sequential manner to get to the root of that problem

- Is the data set okay? More concretely: Is there a lot of noise? Are the features "powerful" enough to discriminate between classes? (It's a good idea to try a bunch of off-the-shelf classifiers to get an initial benchmark; classifiers like random forest, softmax regression, or kernel SVM)
- Did we forget standardizing the features?
- Did we implement and use gradient checking to make sure that our implementation is correct?
- Do we use a random weight initialization scheme (e.g., from a random normal distribution multiplied by a small coefficient < 0) vs. initializing the model parameters to all-zero weights?
- Did we try to increase or decrease the learning rate?
- Have we checked that the cost decreases over time? If yes, have we tried to increase the number of epochs?
- Have we tried to modify the learning rate using momentum learning and/or a decrease constant (e.g., AdaGrad)
- Have we tried different non-linear activation functions other than the one we are currently using (e.g., logistic sigmoid, tanh, or ReLU)?
- Did we try dropout?   
