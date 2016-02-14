# When can a random forest perform terribly?

I'd say any time when your classes are linearly separable by a straight line or hyperplane that is not perpendicular to one of the axes (1). Or if you are also interested in predicting values beyond the training dataset window in a regression problem (2).


(1) The intuition is that decision trees are piece-wise linear functions that partition the feature space perpendicular to the axes. So, instead of drawing a "straight" diagonal line, we get a zig-zag. The same problem occurs with concentric circles and so forth.

(2) The intuition for the regression window is that in decision tree regression, our predicted target variable is the average of the target variables at a terminal node (these come from the training set). So, if the largest value in your training set is "x," we can never make a prediction that is larger than "x," which may be undesirable in certain situations.

A trivial example: Let's say we want to predict the weight of a person (target variable) based on the person's height (feature). We assume the heaviest person in our training set was 180 lbs with a height of 6 ft; the lightest person was 5 ft tall at 150 lbs. Next, let's assume that there's a perfect correlation between height and weight. Eventually, let us  make a prediction for a new data point: We want to predict the weight of a 7 ft person. Using decision tree / random forest regression, your prediction would be max. 180 lbs, which intuitively wouldn't make sense here ...
