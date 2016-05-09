# What is the basic idea behind the dropout technique?

Dropout is a regularization technique, which aims to reduce the complexity of the model with the goal to prevent overfitting.

Using “dropout", you randomly deactivate certain units (neurons) in a layer with a certain probability p from a Bernoulli distribution (typically 50%, but this yet another hyperparameter to be tuned). So, if you set half of the activations of a layer to zero, the neural network won’t be able to rely on particular activations in a given feed-forward pass during training. As a consequence, the neural network will learn different, redundant representations; the network can’t rely on the particular neurons and the combination (or interaction) of these to be present. Another nice side effect is that training will be faster.

Additional technical notes: Dropout is only applied during training, and you need to rescale the remaining neuron activations. E.g., if you set 50% of the activations in a given layer to zero, you need to scale up the remaining ones by a factor of 2. Finally, if the training has finished, you’d use the complete network for testing (or in other words, you set the dropout probability to 0).

For more details, I recommend the original paper: Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* The Journal of Machine Learning Research, 15(1), 1929-1958.
(http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
