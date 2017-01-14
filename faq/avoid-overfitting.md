# How can I avoid overfitting?

In short, the general strategies are to

1. collect more data
2. use ensembling methods that "average" models
3. choose simpler models / penalize complexity

For the first point, it may help to plot learning curves, plotting the training vs. the validation or cross-validation performance. If you see a trend that more data helps with closing the cap between the two, and if you could afford collecting more data, then this would probably the best choice.

In my experience, ensembling is probably the most convenient way to build robust predictive models on somewhat small-sized datasets. As in real life, consulting a bunch of "experts" is usually not a bad idea before making a decision ;).

Regarding the third point, I usually start a predictive modeling task with the simplest model as a benchmark: usually logistic regression. Overfitting can be a real problem if our model has too much capacity — too many model parameters to fit, and too many hyperparameters to tune. If the dataset is small, a simple model is always a good option to prevent overfitting, and it is also a good benchmark for comparison to more "complex" alternatives.    
