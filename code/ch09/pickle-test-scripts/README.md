**Note**

The pickling-section may be a bit tricky so that I included simpler test scripts in this directory (pickle-test-scripts/) to check if your environment is set up correctly. Basically, it is just a trimmed-down version of the relevant sections from Ch08, including a very small movie_review_data subset.

Executing

    python pickle-dump-test.py

will train a small classification model from the `movie_data_small.csv` and create the 2 pickle files 

    stopwords.pkl
    classifier.pkl

Next, if you execute

    python pickle-load-test.py

You should see the following 2 lines as output:

    Prediction: positive
    Probability: 85.71%