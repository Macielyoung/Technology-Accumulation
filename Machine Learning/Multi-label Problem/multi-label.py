
# coding: utf-8

# In[2]:

import scipy
from scipy.io import arff
import pandas as pd
data, meta = scipy.io.arff.loadarff('E://ML//test//yeast-train.arff')
df = pd.DataFrame(data)


# In[5]:

from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split

# this will generate a random multi-label dataset
X, y = make_multilabel_classification(sparse = True, n_labels = 20, return_indicator = 'sparse', allow_unlabeled = False)

# split the dataset into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[6]:

# 1. using binary relevance
# The multi-label problem is broken into some different single class classification problems
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)


# In[7]:

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[8]:

# 2. using classifier chains
# The problem would be transformed into some different single label problems. 
# Different from the previous method, it forms chains in order to preserve label correlation.
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# initialze classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

# accuracy
accuracy_score(y_test, predictions)


# In[9]:

# 3. using label powerset
# The multi-label problem is transformed into a multi-class problem with one multi-class
# classifier is trained on all unique label combinations found in the training data.
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

# accuracy
accuracy_score(y_test,predictions)


# In[10]:

# 4. adapting the algorithm to directly perform multi-label classification
from skmultilearn.adapt import MLkNN

classifier = MLkNN(k = 20)

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

# accuracy
accuracy_score(y_test, predictions)


# In[11]:

# 5. Ensemble method
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
from skmultilearn.ensemble import LabelSpacePartitioningClassifier

# construct base forest classifier
base_classifier = RandomForestClassifier()

# setup problem transformation approach with sparse matrices for random forest
problem_transform_classifier = LabelPowerset(classifier=base_classifier, require_dense=[False, False])

# partition the label space using fastgreedy community detection
# on a weighted label co-occurrence graph with self-loops allowed
clusterer = IGraphLabelCooccurenceClusterer('fastgreedy', weighted=True, include_self_edges=True)

# setup the ensemble metaclassifier
classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

# accuracy
accuracy_score(y_test, predictions)


# In[ ]:

# Case Studies
# 1. Audio Categorization
# 2. Image Categorization
# 3. Bioinformatics
# 4. Text Categorization

