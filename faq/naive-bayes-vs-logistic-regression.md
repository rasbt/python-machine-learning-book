# What is the major difference between naive Bayes and logistic regression?

On a high-level, I would describe it as "generative vs. discriminative" models.

- Generative classifiers learn a model of joint probabilities p(x, y) and use Bayes rule to calculate p(x|y) to make a prediction
- Discriminative models learn the posterior probability p(x|y) "directly"

You can think of discriminative models as "distinguishing between people that speak different languages without actually learning the language".

In discriminative models, you have "less assumptions", e.g,. in naive Bayes and classification, you assume that your p(x|y) follows (typically) a Gaussian, Bernoulli, or Multinomial distribution, and you even violate the assumption of conditional independence of the features. In favor of discriminative models, Vapnik wrote once "one should solve the classification problem directly and never solve a more general problem as an intermediate step".
(Vapnik, Vladimir Naumovich, and Vlamimir Vapnik. Statistical learning theory. Vol. 1. New York: Wiley, 1998.)

I think it really depends on your problem though which method to prefer. I can't find a reference now, but e.g. in classification, naive Bayes converges quicker but has typically a higher error than logistic regression. On small datasets you'd might want to try out naive Bayes, but as your training set size grows, you likely get better results with logistic regression.
