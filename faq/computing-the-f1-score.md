This is an excerpt of an upcoming blog article of mine. Unfortunately, the blog article turned out to be quite lengthy, too lengthy. In the process of pruning, there are hard choices to be made, and this tangent, eh, section needs to go ...
Before I hit the delete button ... maybe this section is useful to others!?

Struggling to continue story where I left off: The "way" we select a model and select amongst different machine learning algorithms all depends on how we evaluate the different models, which in turn depends upon the performance metric we choose. To summarize, the topics we mostly care about are

- estimation of the generalization performance
- algorithm selection
- hyperparameter tuning techniques
- (cross)-validation and sampling techniques
- performance metrics
- class imbalances

But now to the actual section I wanted to share ...

## Interlude: Comparing and Computing Performance Metrics in Cross-Validation -- Imbalanced Class Problems and 3 Different Ways to Compute the F1 Score

Not too long ago, George Forman and Martin Scholz wrote a thought-provoking paper dealing with the comparison and computation of performance metrics across literature, especially when dealing with class imbalances: [Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement (2010)](http://www.hpl.hp.com/techreports/2009/HPL-2009-359.pdf). This is such a nicely written, very accessible paper (and such an important topic)! I highly recommend given this a read. Given that it's not old hat to you, it might change your perspective, the way you read papers, the way you evaluate and benchmark your machine learning models -- and if you decide to publish your results, your readers will benefit as well, that's for sure.

Now, imagine that we want to compare the performance of our new, shiny algorithm to the efforts made in the past. First, we want to make sure that we are comparing "fruits to fruits." Assuming we evaluate on the same dataset, we want to make sure that we use the same cross-validation technique and evaluation metric. I know, this sounds trivial, but we first want to establish this ground rule that we can't compare ROC areas under the curves (AUC) measures to F1 scores ...
On a side note, the use of ROC AUC metrics is still a hot topic of discussion, e.g.,

- JM. Lobo, A. Jiménez-Valverde, and R. Real 2008: [AUC: a misleading measure of the performance of predictive distribution models](http://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2007.00358.x/abstract;jsessionid=40E65D14D4CEEC38F203699F5DCC18C7.f01t03?userIsAuthenticated=false&deniedAccessCustomisedMessage=)
- Jin Huang & C. X. Ling 2005: [Using AUC and accuracy in evaluating learning algorithms](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1388242&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D1388242)
- AP. Bradley 1997 [The use of the area under the ROC curve in the evaluation of machine learning algorithms](http://www.sciencedirect.com/science/article/pii/S0031320396001422)

In any case, let's focus on the F1 score for now summarizing some ideas from Forman & Scholz' paper after defining some of the relevant terminology.

As we probably heard or read before, the F1-score is simply the harmonic mean of precision (PRE) and recall (REC)

F1 = 2 * (PRE * REC) / (PRE + REC)

***What we are trying to achieve with the F1-score metric is to find an equal balance between precision and recall, which is extremely useful in most scenarios when we are working with imbalanced datasets (i.e., a dataset with a non-uniform distribution of class labels).***

---
If we write the two metrics PRE and REC in terms of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), we get:

- PRE = TP / (TP + FP)
- REC = TP / (TP + FN)

Thus, the precision score gives us an idea (expressed as a score from 1.0 to 0.0, from good to bad) of the proportion of how many actual spam emails (TP) we correctly classified as spam among all the emails we classified as spam (TP + FP).
In contrast, the recall (also ranging from 1.0 to 0.0) tells us about how many of the actual spam emails (TP) we "retrieved" or "recalled" (TP + FN).

---

Okay, let's assume we settled on the F1-score as our performance metric of choice to benchmark our new algorithm; coincidentally, the algorithm in a certain paper, which should serve as our reference performance, was also evaluated using the F1 score. Using the same cross-validation technique on the same dataset, this should make this comparison, fair, right? No, no, no, not so fast! On top of choosing the appropriate performance metric -- comparing "fruits to fruits" -- we also have to care about how it's computed in order to compare "apples to apples." This is extremely important if we are comparing performance metrics on imbalanced datasets, which I will explain in a second (based on the results from Forman & Martin Scholz' paper). ***Also, keep in mind that even if our dataset doesn't seem to be imbalanced at first glance, let's think of the Iris dataset with 50 Setosa, 50 Virginica, and 50 Versicolor flowers: What happens if we use a One-vs-Rest (OVR; or One-vs-All, OVA) classification scheme?***

In any case, let's focus on a binary classification problem (a *positive* and a *negative* class) for now using k-fold cross-validation as our cross-validation technique of choice for model selection.

As mentioned before, we calculate the F1 score as

F1 = 2 * (PRE * REC) / (PRE + REC)

Now, what happens if we have a highly imbalanced dataset and perform our k-fold cross validation procedure in the training set? Well, chances are that a particular fold may not contain *a positive* sample so that TP=FN=0. If this doesn't sound too bad, have another look at the recall equation above -- yes, that's a zero-division error! Or, what happens if our classifier predicts the negative class almost all the time (i.e., it has a low false-positive rate)? Again, we get a zero-division error in the precision equation since TP = FP = 0.

What can we do about it? There are two things. Firstly, let's stratify our folds -- stratification means that the random sampling procedure attempts to maintain the class-label proportion across the different folds. Thus, we are unlikely to face problems like having "no samples from the *positive* class" given that *k* is not larger than the number of *positive* samples in the training dataset.

***In practice, different software packages handle the zero-division errors differently: Some don't hesitate throwing run-time exceptions; some may silently substitute the precision and/or recall by a 0 -- make sure what it's doing!*** On top of that, we can compute the F1 score in several distinct ways (and in multi-class problems, we can put the micro- and macro-averaging techniques on top of that, but this is beyond of the scope of this section). As listed by Forman and Scholz, these three different scenarios are

#### (1)

We compute the F1 score for each fold (iteration); then, we compute the average F1 score
from these individual F1 scores.

F1<sub>avg</sub> = 1/k &Sigma;<sup>k</sup><sub>i=1</sub> F1<sup>(i)</sup>


#### (2)

We compute the average precision and recall scores across the *k* folds; then, we use these average scores to compute the final F1 score.

PRE = 1/k &Sigma;<sup>k</sup><sub>i=1</sub> PRE<sup>(i)</sup>

REC = 1/k &Sigma;<sup>k</sup><sub>i=1</sub> REC<sup>(i)</sup>

F1<sub>PRE, REC</sub> = 2 * (PRE * REC) / (PRE + REC)



#### (3)

We compute the number of TP, FP, and FN separately for each fold or iteration, and compute the final F1 score based on these "micro" metrics.

TP = &Sigma;<sup>k</sup><sub>i=1</sub> TP<sup>(i)</sup>

FP = &Sigma;<sup>k</sup><sub>i=1</sub> FP<sup>(i)</sup>

FN = &Sigma;<sup>k</sup><sub>i=1</sub> FN<sup>(i)</sup>

F1<sub>TP, FP, FN</sub> = (2 * TP) / (2 * TP + FP + FN)

(Note that this equation doesn't suffer from the zero-division issue.)

---

Please note that we don't have to worry about the different ways to compute the classification error or accuracy, which we are working with in this blog article aside from this section. The reason is that it doesn't matter whether we we compute the accuracy as


ACC<sub>avg</sub> = 1/k &Sigma;<sup>k</sup><sub>i=1</sub> ACC<sup>(i)</sup>

or

TP = &Sigma;<sup>k</sup><sub>i=1</sub> TP<sup>(i)</sup>

TN = &Sigma;<sup>k</sup><sub>i=1</sub> TN<sup>(i)</sup>

ACC <sub>avg</sub> = (TP + TN) / N

The two approaches are identical, or with a more concrete example: (30 + 40) / 100  = (30/50 + 40/50) / 2 = 0.7.

---

Eventually, Forman and Scholz plaid this game of using different ways to compute the F1 score based on a benchmark dataset with a high-class imbalance (a bit exaggerated for demonstration purposes but not untypical when working with text data). It turns out that the resulting scores (from the identical model) differed substantially:

- F1<sub>avg</sub>: 69%
- F1<sub>PRE, REC</sub>: 73%
- F1<sub>TP, FP, FN</sub>: 58%

***Finally, based on further simulations, Forman and Scholz concluded that the computation of F1<sub>TP, FP, FN</sub> (compared to the alternative ways of computing the F1 score), were yielded the "most unbiased" estimate of the generalization performance using *k*-fold cross-validation.***

In any case, the bottom line is that we should not only choose the appropriate performance metric and cross-validation technique for the task, but we also take a ***closer look at how the different performance metrics are computed in case we cite papers or rely on off-the-shelve machine learning libraries.***
