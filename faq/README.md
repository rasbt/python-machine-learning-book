# Python Machine Learning FAQ

It is always a pleasure to engage in discussions all around machine learning and I am happy to answer any questions regarding this book.

I just thought that it might be worthwhile to compile some documents about
the most commonly asked questions to answer them more thoroughly.

Just drop me your question, feedback, or suggestion via your medium of choice and it will be answered :)

- [mailing List](https://groups.google.com/forum/#!forum/python-machine-learning-book)
- [private email](mailto:mail@sebastianraschka.com)
- [twitter](https://twitter.com/rasbt)
- [GitHub Issue Tracker](https://github.com/rasbt/python-machine-learning-book/issues)

Cheers,
Sebastian

## FAQ

<!--- start -->

### General Questions about Machine Learning and 'Data Science'

- [What are machine learning and data science?](./datascience-ml.md)
- [Why do you and other people sometimes implement machine learning algorithms from scratch?](./implementing-from-scratch.md)
- [What learning path/discipline in data science I should focus on?](./data-science-career.md)
- [At what point should one start contributing to open source?](./open-source.md)
- [How important do you think having a mentor is to the learning process?](./mentor.md)
- [Where are the best online communities centered around data science/machine learning or python?](./ml-python-communities.md)
- [How would you explain machine learning to a software engineer?](./ml-to-a-programmer.md)
- [How would your curriculum for a machine learning beginner look like?](./ml-curriculum.md)
- [What is the Definition of Data Science?](./definition_data-science.md)
- [How do Data Scientists perform model selection? Is it different from Kaggle?](./model-selection-in-datascience.md)

### Questions about the Machine Learning Field

- [How are Artificial Intelligence and Machine Learning related?](./ai-and-ml.md)
- [What are some real-world examples of applications of machine learning in the field?](./ml-examples.md)
- [What are the different fields of study in data mining?](./datamining-overview.md)
- [What are differences in research nature between the two fields: machine learning & data mining?](./datamining-vs-ml.md)
- [How do I know if the problem is solvable through machine learning?](./ml-solvable.md)
- [What are the origins of machine learning?](./ml-origins.md)
- [How was classification, as a learning machine, developed?](./classifier-history.md)
- [Which machine learning algorithms can be considered as among the best?](./best-ml-algo.md)
- [What are the broad categories of classifiers?](./classifier-categories.md)
- [What is the difference between a classifier and a model?](./difference_classifier_model.md)
- [What is the difference between a parametric learning algorithm and a nonparametric learning algorithm?](./parametric_vs_nonparametric.md)
- [What is the difference between a cost function and a loss function in machine learning?](./cost-vs-loss.md)

### Questions about Machine Learning Concepts and Statistics

##### Cost Functions and Optimization

- [Fitting a model via closed-form equations vs. Gradient Descent vs Stochastic Gradient Descent vs Mini-Batch Learning -- what is the difference?](./closed-form-vs-gd.md)
- [How do you derive the Gradient Descent rule for Linear Regression and Adaline?](./linear-gradient-derivative.md)

##### Regression Analysis

- [What is the difference between Pearson R and Simple Linear Regression?](./pearson-r-vs-linear-regr.md)

##### Tree models

- [How does the random forest model work? How is it different from bagging and boosting in ensemble models?](./bagging-boosting-rf.md)
- [What are the disadvantages of using classic decision tree algorithm for a large dataset?](./decision-tree-disadvantages.md)
- [Why are implementations of decision tree algorithms usually binary, and what are the advantages of the different impurity metrics?](./decision-tree-binary.md)
- [Why are we growing decision trees via entropy instead of the classification error?](./decisiontree-error-vs-entropy.md)
- [When can a random forest perform terribly?](./random-forest-perform-terribly.md)

##### Model evaluation

- [What is overfitting?](./overfitting.md)
- [How can I avoid overfitting?](./avoid-overfitting.md)
- [Is it always better to have the largest possible number of folds when performing cross validation?](./number-of-kfolds.md)
- [When training an SVM classifier, is it better to have a large or small number of support vectors?](./num-support-vectors.md)
- [How do I evaluate a model?](./evaluate-a-model.md)
- [What is the best validation metric for multi-class classification?](./multiclass-metric.md)
- [What factors should I consider when choosing a predictive model technique?](./choosing-technique.md)
- [What are the best toy datasets to help visualize and understand classifier behavior?](./clf-behavior-data.md)
- [How do I select SVM kernels?](./select_svm_kernels.md)
- [Interlude: Comparing and Computing Performance Metrics in Cross-Validation -- Imbalanced Class Problems and 3 Different Ways to Compute the F1 Score](./computing-the-f1-score.md)


##### Logistic Regression

- [What is Softmax regression and how is it related to Logistic regression?](./softmax_regression.md)
- [Why is logistic regression considered a linear model?](./logistic_regression_linear.md)
- [What is the probabilistic interpretation of regularized logistic regression?](./probablistic-logistic-regression.md)
- [Does regularization in logistic regression always results in better fit and better generalization?](./regularized-logistic-regression-performance.md)
- [What is the major difference between naive Bayes and logistic regression?](./naive-bayes-vs-logistic-regression.md)
- [What exactly is the "softmax and the multinomial logistic loss" in the context of machine learning?](./softmax.md)
- [What is the relation between Loigistic Regression and Neural Networks and when to use which?](./logisticregr-neuralnet.md)
- [Logistic Regression: Why sigmoid function?](./logistic-why-sigmoid.md)
- [Is there an analytical solution to Logistic Regression similar to the Normal Equation for Linear Regression?](./logistic-analytical.md)

##### Neural Networks and Deep Learning

- [What is the difference between deep learning and usual machine learning?](./difference-deep-and-normal-learning.md)
- [Can you give a visual explanation for the back propagation algorithm for neural networks?](./visual-backpropagation.md)
- [Why did it take so long for deep networks to be invented?](./inventing-deeplearning.md)
- [What are some good books/papers for learning deep learning?](./deep-learning-resources.md)
- [Why are there so many deep learning libraries?](./many-deeplearning-libs.md)
- [Why do some people hate neural networks/deep learning?](./deeplearning-criticism.md)
- [How can I know if Deep Learning works better for a specific problem than SVM or random forest?](./deeplearn-vs-svm-randomforest.md)
- [What is wrong when my neural network's error increases?](./neuralnet-error.md)
- [How do I debug an artificial neural network algorithm?](./nnet-debugging-checklist.md)
- [What is the difference between a Perceptron, Adaline, and neural network model?](./diff-perceptron-adaline-neuralnet.md)
- [What is the basic idea behind the dropout technique?](./dropout.md)

##### Other Algorithms for Supervised Learning

- [Why is Nearest Neighbor a Lazy Algorithm?](./lazy-knn.md)

##### Unsupervised Learning

- [What are some of the issues with clustering?](./issues-with-clustering.md)

##### Semi-Supervised Learning

- [What are the advantages of semi-supervised learning over supervised and unsupervised learning?](./semi-vs-supervised.md)

##### Ensemble Methods

- [Is Combining Classifiers with Stacking Better than Selecting the Best One?](./logistic-boosting.md)

##### Preprocessing, Feature Selection and Extraction

- [Why do we need to re-use training parameters to transform test data?](./scale-training-test.md)
- [What are the different dimensionality reduction methods in machine learning?](./dimensionality-reduction.md)
- [What is the difference between LDA and PCA for dimensionality reduction?](./lda-vs-pca.md)
- [When should I apply data normalization/standardization?](./when-to-standardize.md)
- [Does mean centering or feature scaling affect a Principal Component Analysis?](./pca-scaling.md)
- [How do you attack a machine learning problem with a large number of features?](./large-num-features.md)
- [What are some common approaches for dealing with missing data?](./missing-data.md)
- [What is the difference between filter, wrapper, and embedded methods for feature selection?](./feature_sele_categories.md)
- [Should data preparation/pre-processing step be considered one part of feature engineering? Why or why not?](./dataprep-vs-dataengin.md)
- [Is a bag of words feature representation for text classification considered as a sparse matrix?](./bag-of-words-sparsity.md)
- [How can I apply an SVM to categorical data?](./svm_for_categorical_data.md)

##### Naive Bayes

- [Why is the Naive Bayes Classifier naive?](./naive-naive-bayes.md)
- [What is the decision boundary for Naive Bayes?](./naive-bayes-boundary.md)
- [Is it possible to mix different variable types in Naive Bayes, for example, binary and continues features?](./naive-bayes-vartypes.md)

##### Other

- [What is Euclidean distance in terms of machine learning?](./euclidean-distance.md)
- [When should one use median, as opposed to the mean or average?](./median-vs-mean.md)

##### Programming Languages and Libraries for Data Science and Machine Learning

- [Is R used extensively today in data science?](./r-in-datascience.md)
- [What is the main difference between TensorFlow and scikit-learn?](./tensorflow-vs-scikitlearn.md)

<!--- end -->

<br>
<br>
<br>




### Questions about the Book

- [Can I use paragraphs and images from the book in presentations or my blog?](./copyright.md)
- [How is this different from other machine learning books?](./different.md)
- [Which version of Python was used in the code examples?](./py2py3.md)
- [Which technologies and libraries are being used?](./technologies.md)
- [Which book version/format would you recommend?](./version.md)
- [Why did you choose Python for machine learning?](./why-python.md)
- [Why do you use so many leading and trailing underscores in the code examples?](./underscore-convention.md)
- [What is the purpose of the `return self` idioms in your code examples?](./return_self_idiom.md)
- [Are there any prerequisites and recommended pre-readings?](./prerequisites.md)
