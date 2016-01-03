# What are some common approaches for dealing with missing data?

Many different approaches exist for dealing with missing values; I'd roughly categorize our options into a) deletion and b) imputation techniques.

## a) Deletion

1) We have a lot of training samples and can afford deleting some of those. Here, we can simply remove samples with missing feature values from the dataset entirely.

2) We have a large number of feature columns and some of them are redundant. Relatively many samples have a missing feature value in a certain column. In this scenario, it may be a good idea to remove these feature columns with missing values entirely.
 
## b) Imputation

If we can't afford deleting data points, we could use imputation techniques to "guess" placeholder values from the remaining data points.

1) The simplest imputation technique may be the replacement of a missing feature value by its feature column's mean (median or mode).

2) Instead of replacing a feature value by its column mean, we can only consider the k-nearest neighbors of this datapoint for computing the mean (median or mode) -- we identify the neighbors based on the remaining feature columns that don't have missing values.
