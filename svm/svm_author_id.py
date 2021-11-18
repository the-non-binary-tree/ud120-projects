#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
### your code goes here ###
clf = svm.SVC(
    C=10000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False
)
t0 = time()
clf.fit(features_train, labels_train)
print('Training time:', round(time() - t0, 3), 's')

t0 = time()
pred = clf.predict(features_test)
print('Predicting time:', round(time() - t0, 3), 's')

accuracy = accuracy_score(labels_test, pred)
print(accuracy)

# for i in [10, 26, 50]:
#     i_pred = pred[i]
#     print(f'Prediction at index {i}: {i_pred}')

print(f'Chris email predictions: {np.count_nonzero(pred == 1)}')
#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
