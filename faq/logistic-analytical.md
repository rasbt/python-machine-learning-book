# Is there an analytical solution to Logistic Regression similar to the Normal Equation for Linear Regression?

Unfortunately, there is no closed-form solution for maximizing the log-likelihood (or minimizing the inverse, the logistic cost function); at least it has not been found, yet.

There's the exception  where you only have 2 obervations, and there is this paper

*Lipovetsky, Stan. ["Analytical closed-form solution for binary logit regression by categorical predictors."](http://www.tandfonline.com/doi/abs/10.1080/02664763.2014.932760) Journal of Applied Statistics 42.1 (2015): 37-49. (Analytical closed-form solution for binary logit regression by categorical predictors)*

which "shows that for categorical explanatory variables, it is possible to present the solution in the analytical closed-form formulae."
The problem is that the logistic sigmoid function is non-linear -- in case of linear regression, you  are assuming independent Gaussian noise.
