# Should data preparation/pre-processing step be considered one part of feature engineering? Why or why not?

I think there's a fuzzy boundary between these two areas of tasks. I see data preparation more as a technical/computational task. E.g., if you think about getting the data into the "right" format, choosing the appropriate data structure / database, and so forth.
Then, there's data cleaning, which can also be grouped into the "preparation / pre-processing" category. Here, you may want to think about detecting duplications, how to deal with outliers, and how to deal with missing data.

To me, feature engineering is a bit different. I see it more as a "data/feature creation" step rather than a data "sanitizing" step. Feature engineering may include all different sorts of feature transformations in both directions: Higher-dimensional feature spaces (e.g., polynomials), lower dimensional feature spaces (dimensionality reduction like PCA, LDA, etc., hashing, clustering), or you keep the dimensions but change the distribution of your data (e.g., log transformation, standardization, min-max scaling etc.)
