# Why is logistic regression considered a linear model?


The short answer is: Logistic regression is considered a generalized linear model because the outcome **always** depends on the **sum** of the inputs and parameters. Or in other words, the output cannot depend on the product (or quotient, etc.) of its parameters!

So, why is that? Let’s recapitulate the basics of logistic regression first, which hopefully makes things more clear. Logistic regression is an algorithm that learns a model for binary classification. A nice side-effect is that it gives us the *probability* that a sample belongs to class 1 (or vice versa: class 0). Our objective function is to minimize the so-called logistic function &Phi; (a certain kind of sigmoid function); it looks like this:

![](./logistic_regression_linear/2.png)

Now, if *&phi;(z)* is larger than *0.5* (alternatively: if *z* is larger than *0*), we classify an input as class 1 (and class 0, otherwise).  Although logistic regression produces a linear decision surface (see the classification example in the figure below) this logistic (activation) function doesn't look very linear at all, right!?doesn't look very linear at all, right!?

![](./logistic_regression_linear/4.png)


So, let's dig a bit deeper and take a look at the equation we use to compute *z* -- the net input function!

![](./logistic_regression_linear/1.png)

The net input function is simply the dot product of our input features and the respective model coefficients **w**:

![](./logistic_regression_linear/3.png)

Here, x<sub>0</sub> refers to the weight of the bias unit which is always equal to 1 (a detail we don’t have to worry about here). I know, mathematical equations can be a bit "abstract" at times, so let's look at a concrete example.

Let's assume we have a sample training point **x** consisting of 4 features (e.g., *sepal length*, *sepal width*, *petal length*, and *petal width* in the [*Iris dataset*](https://archive.ics.uci.edu/ml/datasets/Iris)):

    x = [1, 2, 3, 4]

Now, let's assume our weight vector looks like this:

    w = [0.5, 0.5, 0.5, 0.5]

Let's compute *z* now!

z = w<sup>T</sup>x = 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 5

---

Not that it is important, but we have a 99.3% chance that this sample belongs to class 1:

*&Phi;(z=148.41) = 1 / (1 + e<sup>-5</sup>) = 0.993*

---

The key is that our model is ***additive***
our outcome *z* depends on the additivity of the weight parameter values, e.g., :

*z = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>*

There's no interaction between the weight parameter values, nothing like w<sub>1</sub>x<sub>1</sub> * w<sub>2</sub>x<sub>2</sub>

 or so, which would make our model non-linear!
