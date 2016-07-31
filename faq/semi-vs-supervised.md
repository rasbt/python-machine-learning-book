# What are the advantages of semi-supervised learning over supervised and unsupervised learning?

Obviously, we are working with a labeled dataset when we are building (typically predictive) models using supervised learning. The goal of unsupervised learning is often of exploratory nature (clustering, compression) while working with unlabeled data.

In semi-supervised learning, we are trying to solve a supervised learning approach using labeled data augmented by unlabeled data; the number of unlabeled or partially labeled samples is often larger than the number of labeled samples, since the former are less expensive and easier to obtain. So, our goal is to overcome one of the problems of supervised learning -- having not enough labeled data. Adding cheap and abundant unlabeled data, we are hoping to build a better model than using supervised learning alone.

Although semi-supervised learning sounds like a powerful approach, we have to be careful. Semi-supervised learning is not always "the hammer to the nail" that we are looking for -- sometimes it works great, sometimes it doesn't. Here's a great paper on this:

Singh, Aarti, Robert Nowak, and Xiaojin Zhu. "[Unlabeled data: Now it helps, now it doesn't.](http://www.cs.cmu.edu/~aarti/pubs/NIPS08_ASingh.pdf)" Advances in Neural Information Processing Systems. 2009.

Also, we have to keep in mind that we need to make certain assumptions (manifold, cluster, or smoothness assumptions; see here for more details: [Semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning#Assumptions_used_in_semi-supervised_learning)) when we are using semi-supervised algorithms and have to make sure that they are not violated.
