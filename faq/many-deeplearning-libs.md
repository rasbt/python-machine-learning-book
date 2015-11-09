#  Why are there so many deep learning libraries?

In my opinion, the main reason is that everything in the Deep Learning field is still highly experimental. The libraries are usually a side-product of someone's own research. In addition, deep learning algorithms are not so general (in terms of writing a general code that can be applied to many different problems) in comparison to other algorithms, e.g., Random forests, logistic regression, SVMs, etc. Thus, everyone has his or her particular idea how a good interface may look like; hence, they may end up developing a new library. Also, they probably want to incorporate their personal research in the respective library and get credit for it -- it's much easier to have your personal library that you can tweak and change as you wish.

However, I also think that it may only seem that there are so many libraries because it is a truly trendy topic at the moment, and we are living in the time and age where open-source and code sharing is (fortunately) very popular. I guess there are 100x more libraries that implement SVMs, logistic regression etc. than deep learning libraries.


Anyway, if you are interested in implementing neural networks yourself, have a look at [Theano](http://deeplearning.net/software/theano/) -- NumPy on steroids as how it is commonly called. Not only does it allow you to use numerical expressions more efficiently, but they also implement tensors and let you utilize GPUs. Theano is actually what most of the "many deep learning libraries" in Python are using, e.g,.

- [Lasagne](https://github.com/Lasagne/Lasagne)
- [Keras](http://keras.io)
- [PyLearn 2](https://github.com/lisa-lab/pylearn2)

...
