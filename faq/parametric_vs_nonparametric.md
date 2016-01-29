# What is the difference between a parametric learning algorithm and a nonparametric learning algorithm?

The term "non-parametric" might sound a bit confusing at first: non-parametric does not mean that they have NO parameters! On the contrary, non-parametric models (can) become more and more complex with an increasing amount of data.

So, in a parametric model, we have a finite number of parameters, and in nonparametric models, the number of parameters is (potentially) infinite. Or in other words, in nonparametric models, the complexity of the model grows with the number of training data; in parametric models, we have a fixed number of parameters (or a fixed structure if you will).

Linear models such as linear regression, logistic regression, and linear Support Vector Machines are typical examples of a parametric "learners;" here, we have a fixed size of parameters (the weight coefficient.) In contrast, K-nearest neighbor, decision trees, or RBF kernel SVMs are considered as non-parametric learning algorithms since the number of parameters grows with the size of the training set. --  K-nearest neighbor and decision trees, that makes sense, but why is an RBF kernel SVM non-parametric whereas a linear SVM is parametric? In the RBF kernel SVM, we construct the kernel matrix by computing the pair-wise distances between the training points, which makes it non-parametric.

In the field of statistics, the term parametric is also associated with a specified probability distribution that you "assume" your data follows, and this distribution comes with the finite number of parameters (for example, the mean and standard deviation of a normal distribution); you don't make/have these assumptions
in non-parametric models. So, in intuitive terms, we can think of a non-parametric model as a "distribution" or (quasi) assumption-free model.

However, keep in mind that the definitions of "parametric" and "non-parametric" are "a bit ambiguous" at best; according to the "The Handbook of Nonparametric Statistics 1 (1962) on p. 2:
“A precise and universally acceptable definition of the term ‘nonparametric’ is not presently available. The viewpoint adopted in this handbook is that a statistical procedure is of a nonparametric type if it has properties which are satisfied to a reasonable approximation when some assumptions that are at least of a moderately general nature hold.”
