# Is a bag of words feature representation for text classification considered as a sparse matrix?

It depends on your vocabulary and dataset, but typically: Yes, definitely!


By definition, a sparse matrix is called "sparse" if most of its elements are zero. In the bag of words model, each document is represented as a word-count vector. These counts can be binary counts (does a word occur or not) or absolute counts (term frequencies, or normalized counts), and the size of this vector is equal to the number of elements in your vocabulary. Thus, if most of your feature vectors are sparse, our bag-of-words feature matrix is most likely sparse as well!


Now, the question is: "When are these feature vectors sparse?" It reallly depends on the size of our vocabulary, and the length and variety of the documents in our training corpus. For instance, the shorter and more similar the documents in our training set are, the more likely it is that we end up with a dense matrix, ---- it's still very unlikely in practice, though!

Here's a trivial example ... Let's suppose we have 3 documents:


- Doc1: Hello, World, the sun is shining
- Doc2: Hello world, the weather is nice
- Doc3: Hello world, the wind is cold


Then, our vocabulary would look like this (using 1-grams without stop word removal):



Vocabulary: [hello, world, the, wind, weather, sun, is, shining, nice, cold]


The corresponding, binary feature vectors are:


- Doc1: [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
- Doc2: [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
- Doc3: [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]


Which we use to construct the dense matrix:

[[1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
[1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
[1, 1, 1, 1, 0, 0, 1, 0, 0, 1] ]


As we can see, we have 17 x 1 and 13 x 0; so, by definition, this wouldn't be a sparse matrix. However, we can also guess how unlikely this scenario is in a real-world application ;)   
