# What is the difference between a classifier and a model?

Essentially, the terms "classifier" and "model" are synonymous in certain contexts; however, sometimes people refer to "classifier" as the learning algorithm that learns the model from the training data. To makes things more tractable, let's define some of the key terminology:

- ***Training sample:*** A training sample is a data point *x* in an available training set that we use for tackling a predictive modeling task. For example, if we are interested in classifying emails, one email in our dataset would be one training sample. Sometimes, people also use the synonymous terms *training instance* or *training example*.

- ***Target function:*** In predictive modeling, we are typically interested in modeling a particular process; we want to learn or approximate a particular function that, for example, let's us distinguish spam from non-spam email. The ***target function*** *f(x) = y* is the **true** function *f* that we want to model.

- ***Hypothesis:*** A hypothesis is a certain function that we believe (or hope) is similar to the true function, the *target function* that we want to model. In context of email spam classification, it would be the *rule* we came up with that allows us to separate spam from non-spam emails.

- ***Model:*** In machine learning field, the terms *hypothesis* and *model* are often used interchangeably. In other sciences, they can have different meanings, i.e., the hypothesis would be the "educated guess" by the scientist, and the *model* would be the manifestation of this *guess* that can be used to test the hypothesis.

- ***Learning algorithm:*** Again, our goal is to find or approximate the ***target function***, and the learning algorithm is a set of instructions that tries to *model* the target function using our training dataset. A learning algorithm comes with a ***hypothesis space***, the set of possible hypotheses it can come up with in order to model the unknown target function by formulating the *final hypothesis*

- ***Classifier:*** A classifier is a special case of a *hypothesis* (nowadays, often learned by a machine learning algorithm). A *classifier* is a *hypothesis* or *discrete-valued function* that is used to assign (categorical) class labels to particular data points. In the email classification example, this classifier could be a hypothesis for labeling emails as spam or non-spam. However, a *hypothesis* must not necessarily be synonymous to a *classifier*. In a different application, our *hypothesis* could be a function for mapping study time and educational backgrounds of students to their future SAT scores.

So, we can say that a *classifier* is a special case of a *hypothesis* or *model*: a classifier is a function that assigns a class label to a data point.
