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
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection





#
#train_file = '../S1/D6_26/8dialects/train'  
#test_file = '../S1/D6_26/8dialects/dev'
train_file = '../data/D6_26/6dialects/splited_train/BEI'  
test_file = '../data/D6_26/6dialects/splited_dev/BEI'

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


def basic_tokenize(tweet):
    return tweet.split(' ')

def skipgram_tokenize(tweet, n=None, k=None, include_all=True):
    from nltk.util import skipgrams
    tokens = [w for w in basic_tokenize(tweet)]
    if include_all:
        result = []
        for i in range(k+1):
            skg = [w for w in skipgrams(tokens, n, i)]
            result = result+skg
    else:
        result = [w for w in skipgrams(tokens, n, k)]
    result=set(result)
    #print(result)
    return result

def make_skip_tokenize(n, k, include_all=True):
    return lambda tweet: skipgram_tokenize(tweet, n=n, k=k, include_all=include_all)

#data_train_size_mb = size_mb(data_train.data)
#data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(target_names))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target
l_acc = []
print("Extracting features from the training data using a sparse vectorizer")
#t0 = time()
#opts.use_hashing = True
#6
union = FeatureUnion([
                    ("w_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,3))),
                    ("c_wb", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(2,5))),
                    ("c_wb5", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(2,5) )),
                    ("sk",TfidfVectorizer(sublinear_tf=True, max_df=0.5,tokenizer=make_skip_tokenize(n=2, k=2)))
                      ],
#transformer_weights={
#            'w_v': 0.8,
#            'c_wb': 0.9,
#           ' c_wb5':0.5,
#            'sk': 0.4,
#        }
#,
)
#union.fit_transform(data_train.data)
X_train = union.fit_transform(data_train.data) #union.fit_transform(data_train.data)
#Y_train = union.transform
X_test = union.transform(data_test.data)

print("Combined space has", X_train.shape[1], "features")

# this is for lev only
#svm = SGDClassifier(alpha=0.001, max_iter=50,penalty="l2")
# this is for high level lev, msa, eg, na
estimators = []
#sgd = SGDClassifier(alpha=0.00001, max_iter=50,penalty="l2") 
#estimators.append(('sgd', sgd))

svc = LinearSVC(penalty='l2', dual=False,tol=1e-3)
estimators.append(('svc',svc))
mnb= MultinomialNB(alpha=.01)
estimators.append(('mnb',mnb))
bnb= BernoulliNB(alpha=.01)
estimators.append(('bnb',bnb))

kfold= 10
ensemble = VotingClassifier(estimators)
#results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
#print(results.mean())






ensemble.fit(X_train, y_train)
#pipeline = Pipeline([("features", union), ("svm", svm)])

pred = ensemble.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, pred,target_names=target_names))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred))