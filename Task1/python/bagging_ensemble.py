#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:08:25 2019

@author: xabuka
"""

# best Identify langauge code


from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection





train_file = '../../MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/Dialect6/Training/post_clean'
    
test_file = '../../MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/Dialect6/Developing/post_clean'

print("Loading MADAR dataset for categories:")

data_train = load_files(train_file, encoding = 'utf-8',decode_error='ignore')

data_test = load_files(test_file, encoding = 'utf-8',decode_error='ignore')


#X_train = data_train.data
y_train = data_train.target
#X_test = data_test.data
y_test = data_test.target
print("Loading MADAR dataset for categories:")
print(data_train.target_names)
# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names



print("Traing Data:   {0}".format(len(data_train.data)))
print("Testing Data:   {0}".format(len(data_test.data)))


#word_vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,1))
char_vect  = TfidfVectorizer(max_features = 50000, sublinear_tf=True,norm ='l1', max_df=0.75,analyzer = 'char_wb', ngram_range=(2,5))


union = FeatureUnion([("w_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,2)
                                 )),
                       ("c_wb", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(2,5)
                                 )),
#                       ("c_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(2,5)
#                                 ))
                       ])

#union.fit_transform(data_train.data)
X_train = union.fit_transform(data_train.data) #union.fit_transform(data_train.data)
#Y_train = union.transform
X_test = union.transform(data_test.data)

print("Combined space has", X_train.shape[1], "features")

# this is for lev only
#svm = SGDClassifier(alpha=0.001, max_iter=50,penalty="l2")
# this is for high level lev, msa, eg, na

svm = MultinomialNB(alpha=.01)#SGDClassifier(alpha=0.00001, max_iter=50,penalty="l2") 
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = BaggingClassifier(base_estimator=svm, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())




model.fit(X_train, y_train)
#pipeline = Pipeline([("features", union), ("svm", svm)])

pred = model.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, pred,target_names=target_names))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred))