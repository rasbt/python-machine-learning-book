# When training an SVM classifier, is it better to have a large or small number of support vectors? 


Unfortunately, like so often in machine learning applications, it really depends on the dataset. If we train an RBF kernel SVM, we will typically end up with more support vectors than in a linear model. If our data is linearly separable, the latter may be better, and if that's not the case, the former may be better.

Also, we have to differentiate between computational efficiency and generalization performance. If we increase the number of support vectors, our classification becomes more "expensive", especially in kernel SVM where we have to recalculate the distances between every new sample and the entire training set.


I'd say the best way to tackle the predictive performance problem is simply to evaluate the model, plot learning curves, do k-fold and/or cross validation, and see what works best on our given dataset.   
