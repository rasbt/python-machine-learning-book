Sebastian Raschka, 2015

# Python Machine Learning - Supplementary Datasets

## MNIST Dataset

- Used in chapters 12 and 13


The MNIST dataset was constructed from two datasets of the US National Institute of Standards and Technology (NIST). The training set consists of handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. Note that the test set contains handwritten digits from different people following the same split.

**Features**

Each feature vector (row in the feature matrix) consists of 784 pixels (intensities) -- unrolled from the original 28x28 pixels images.

- Number of samples: A subset of 5000 images (the first 500 digits of each class)

- Target variable (discrete): {500x 0, ..., 500x 9}


### References

- Source: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Y. LeCun and C. Cortes. Mnist handwritten digit database. AT&T Labs [Online]. Available: http://yann. lecun. com/exdb/mnist, 2010.


### Loading MNIST

- The description and code from [chapter 12](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch12/ch12.ipynb#Obtaining-the-MNIST-dataset)

In addition, I added to convenience function to one of my external machine learning packages

- [A function that loads the MNIST dataset into NumPy arrays](http://rasbt.github.io/mlxtend/user_guide/data/load_mnist/)
- [A utility function that loads the MNIST dataset from byte-form into NumPy arrays](http://rasbt.github.io/mlxtend/user_guide/data/mnist_data/)
