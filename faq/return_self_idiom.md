# What is the purpose of the `return self` idioms in your code examples?

Many (if not all) of the object-oriented implementations in my code examples return `self` in their respective calls -- and scikit-learn does this, too! So, what is the rational behind a method that returns the the object itself? The answer is a rather simple one: "Chaining," which enables us to concatenate operations more conveniently (and efficiently) by feeding the answer of an operations into the next.

For example, an implementation such as

    class Perceptron(object):
       def __init__(self, …):
           …

       def fit(self, …):
           return self

       def predict(self, …):
           return self

would allow us to write a compact notation such as

    prediction = Perceptron().fit(X, y).predict(X)

in one line of code instead of

    p = Perceptron()
    p.fit(X, y)
    prediction = p.predict(X)
