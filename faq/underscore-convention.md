# Why do you use so many leading and trailing underscores in the code examples?

I just received a question from a reader who asked me about the usage of underscores in the Python `class`es. I totally agree that it may look a bit weird to write

    self.gamma_ = 'some value'

instead of just

    self.gamma = 'some value'

Or

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

instead of

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

The short answer is, the trailing underscore (`self.gamma_`) in class attributes is a scikit-learn convention to denote "estimated" or "fitted" attributes.
The leading underscores are (`_sigmoid(self, z)`) denote private methods that the user should not bother with.


In brief: As a reader, you can safely ignore those underscores, however, if you are curious about their intention, please read on!

## Leading underscores in class methods

The usage of underscores for naming class methods is a common Python convention to distinguish between private and public methods. Basically, you do not want the user to worry about these private methods, which is why they do not appear in the help menu.

    class MyClass(object):
        def __init__(self, param='some_value'):
            pass

        def public(self):
            'User, this public method is for you!'
            return 'public method'

        def _indicate_private(self):
            return 'private method'

        def __pseudo_private(self):
            return 'really private method'

<br>

    >>> help(MyClass)

    Help on class MyClass in module __main__:

    class MyClass(builtins.object)
     |  Methods defined here:
     |  
     |  __init__(self, param='some_value')
     |  
     |  public(self)
     |      User, this public method is for you!
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)


   * Note that `__init__` is an exception, `__init__` is a special method that is required to initialize a class.

Let us initialize a new object and call this "public" class:

    >>> MyObj = MyClass()
    >>> MyObj.public()
    'public method'

The single underscore in `_indicate_private` indicates privacy. Typically, private methods are helper methods to facilitate refactoring, encapsulation, unit testing, and so on. Basically, it helps the developers to write cleaner and more efficient code. Although these private methods are hidden from the help menu, we can still call them directly just like the `public methods`:


    >>> MyObj._indicate_private()
    'private method'

- Please keep in mind that calling private methods is at your own risk; the developers usually take no responsibilities for odd things that may happen if you call private methods as a user.

The indication of "privacy" is a bit stronger if we use 2 preceding underscores, for example, calling the `__pseudo_private` method directly like a regular method does not work anymore:

    >>> MyObj.__pseudo_private()
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    <ipython-input-34-80b1e62564e1> in <module>()
    ----> 1 MyObj.__pseudo_private()

    AttributeError: 'MyClass' object has no attribute '__pseudo_private'

To call the a private methods that is prefaced with 2 underscores, we need to adhere to the "name mangling" rules; that is, we need to add a `_classname` prefix, to call the method, for example,

    >>> MyObj._MyClass__pseudo_private()
    'really private method'



## Class attributes with trailing underscores

In contrast to the leading underscore, the trailing underscores in class attributes do not have any "technical" effects. In fact, this is just a convention that I adopted from scikit-learn out of habit.

Here are two excerpts from the scikit-learn [developer/contributor documentation](http://scikit-learn.org/stable/developers/):

> Attributes that have been estimated from the data must always have a name ending with trailing underscore `_`, for example, the coefficients of some regression estimator would be stored in a `coef_` attribute after `fit()` has been called.


> Also it is expected that parameters with trailing underscore `_` are not to be set inside the ``__init__`` method. All and only the public attributes set by `fit()` have a trailing `_`. As a result the existence of parameters with trailing `_` is used to check if the estimator has been fitted.

To see it in action, let us create a primitive `Estimator`:

    class MyEstimator():
        def __init__(self):
            self.param = 1.0

        def fit(self):
            self.fit_param_ = 0.1

Intuitively, attributes that are in `__init__` are accessible after we initialized a new object:

    >>> est = MyEstimator()
    >>> est.param
    1.0

unlike the parameters that are instanciated in the `fit` method:

    >>> est.fit_param_
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    <ipython-input-48-c6cc3a403892> in <module>()
    ----> 1 est.fit_param_

    AttributeError: 'MyEstimator' object has no attribute 'fit_param_'

As expected, the `fit` attributes are then "available" after we called the `fit` method:

    >>> est.fit()
    est.fit_param_
    0.1
