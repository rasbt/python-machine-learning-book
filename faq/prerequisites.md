# Are There Any Prerequisites and Recommended Pre-Readings?


Sure, book goes a bit deeper into the mathematical foundations of certain learning algorithms than other "practical" Python machine learning books, but it really doesn't require a strong math background. I tried to make this journey as smooth as possible by building on concepts incrementally.


Let me give you a concrete example, and please don't worry if this jargon does not make sense to you (yet). Here comes a more verbatim illustration of what I mean when I say "we are going through this together, one step at a time:" Chapter 2 introduces the early perceptron algorithm, and from there, we will learn about convex optimization using a simple, related algorithm called Adaline (short for Adaptive Linear Neuron). Traveling through time and looking at the early beginnings of machine learning, we build our knowledge step by step. Using the Adaline example, we will learn about one of the most popular and fundamental optimization algorithms, gradient descent, including related concepts such as stochastic gradient descent, on-line learning, and mini-batch learning. In the following chapter, Chapter 3, we will build on the concepts of Adaline and learn how two of the most popular off-the-shelf algorithms work, logistic regression and support vector machines. Moreover, we will contrast these algorithms to many other popular learning algorithms such as decision trees, random forests, K-nearest neighbors, and so on. After we developed good habits for preprocessing our training data, evaluating our models, and looking at regression analysis and clustering, we will come back to our Adaline and implement multi-layer artificial neural networks. These will come with all the bells and whistles we learned about in previous chapters such as regularization to prevent overfitting, mini-batch learning, adaptive learning rates, sigmoid/logistic activation functions, multi-class classification, and many more.


Unfortunately, teaching the fundamentals of Python was a little beyond the scope of the book -- it's 454 pages long after all. Although I tried my best to explain the code examples, it probably wouldn't hurt to be somewhat familiar with NumPy and matplotlib. Don't worry about scikit-learn and Theano, really.
**Please keep in mind that this book is about machine learning augmented by practical Python code example that help illustrate the learned material but also give you something that you can take into the real world to apply to your own projects.**
Please don't hesitate to write me a [mail](mailto:mail@sebastianraschka.com) if you are stuck at a particular concept or code example. In addition to the excellent, official NumPy, pandas, and matplotlib tutorials, let me give me two more resources that may come in handy if you haven't worked with Python's sci-stack yet and want some additional background knowledge:


- [Scipy Lecture Notes - One document to learn numerics, science, and data with Python](http://www.scipy-lectures.org)
- [Zico Kolter's Linear Algebra Review and Reference](http://cs229.stanford.edu/section/cs229-linalg.pdf)


Two great resources, although jargon may sound daunting sometimes, it's really that simple. :)   
