import pickle
import os
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)


df = pd.read_csv('./movie_data_small.csv', encoding='utf-8')

#df.loc[:100, :].to_csv('./movie_data_small.csv', index=None)


X_train = df['review'].values
y_train = df['sentiment'].values

X_train = vect.transform(X_train)
clf.fit(X_train, y_train)

pickle.dump(stop,
            open('stopwords.pkl', 'wb'),
            protocol=4)

pickle.dump(clf,
            open('classifier.pkl', 'wb'),
            protocol=4)
