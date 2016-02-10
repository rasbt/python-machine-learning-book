# Why do we need to re-use training parameters to transform test data?

Many machine learning algorithms require that features are on the same scale; for example, if we compute distances such as in nearest neighbor algorithms. Also, optimization algorithms such as gradient descent work best if our features are centered at mean zero with a standard deviation of one — i.e., the data has the properties of a standard normal distribution. One of the few categories of machine algorithms that are truly scale invariant are the tree-based methods.

Now, a commonly asked question is how we scale our dataset correctly. For simplicity, I will write the examples in pseudo code using the “standardization” procedure. However, note that the same principles apply to other scaling methods such as min-max scaling.

In practice, I’ve seen many ways for scaling a dataset prior to feeding it to a learning algorithm. Can you guess which one is “correct?" 

### Scenario 1:

    scaled_dataset = (dataset - dataset_mean) / dataset_std_deviation
    
    train, test = split(scaled_dataset)

### Scenario 2:

    train, test = split(dataset)

    scaled_train =  (train - train_mean) / train_std_deviation

    scaled_test = (test - test_mean) / test_std_deviation

### Scenario 3:

    scaled_train =  (train - train_mean) / train_std_deviation

    scaled_test = (test - train_mean) / train_std_deviation

That’s right, the “correct” way is *Scenario 3*. I agree, it may look a bit odd to use the training parameters and re-use them to scale the test dataset. (Note that in practice, if the dataset is sufficiently large, we wouldn’t notice any substantial difference between the scenarios 1-3 because we assume that the samples have all been drawn from the same distribution.)

Again, why *Scenario 3*? The reason is that we want to pretend that the test data is "new, unseen data.” We use the test dataset to get a good estimate of how our model performs on any new data.

Now, in a real application, the new, unseen data could be just 1 data point that we want to classify. (How do we estimate mean and standard deviation if we have only 1 data point?) That's an intuitive case to show why we need to keep and use the training data parameters for scaling the test set.

To recapitulate: If we standardize our training dataset, we need to keep the parameters (*mean* and *standard deviation* for each feature). Then, we'd use these parameters to transform our test data and any future data later on

Let me give a hands-on example why this is important!

Let's imagine we have a simple training set consisting of 3 samples with 1 feature column (let's call the feature column “length in cm"):

- sample1: 10 cm -> class 2
- sample2: 20 cm -> class 2
- sample3: 30 cm -> class 1

Given the data above, we compute the following parameters:

- mean: 20
- standard deviation: 8.2

If we use these parameters to standardize the same dataset, we get the following values:

- sample1: -1.21 -> class 2
- sample2: 0 -> class 2
- sample3: 1.21 -> class 1

Now, let's say our model has learned the following hypotheses: It classifies samples with a standardized length value < 0.6 as class 2 (class 1 otherwise). So far so good. Now, let’s imagine we have 3 new unlabeled data points that you want to classify.

- sample4: 5 cm -> class ?
- sample5: 6 cm -> class ?
- sample6: 7 cm -> class ?

If we look at the "unstandardized “length in cm" values in our training datast, it is intuitive to say that all of these samples are likely belonging to class 2. However, if we standardize these by re-computing the *standard deviation* and and *mean* from the new data, we would get similar values as before (i.e., properties of a standard normal distribtion) in the training set and our classifier would (probably incorrectly) assign the “class 2” label to the samples 4 and 5.

- sample5: -1.21 -> class 2
- sample6: 0 -> class 2
- sample7: 1.21 -> class 1

However, if we use the parameters from your "training set standardization, we will get the following standardized values

- sample5: -18.37
- sample6: -17.15 
- sample7: -15.92

Note that these values are more negative than the value of sample1 in the original training set, which makes much more sense now!