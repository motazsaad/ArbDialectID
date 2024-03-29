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
import matplotlib.pyplot as plt
import seaborn as sns 


#92.6 with post and pre clean



#
#train_file = '../S1/D6_26/train'
#    
#test_file = '../S1/D6_26/dev'
train_file = '../data/Dialect6/Multi_data/train/pre_clean'
    
test_file = '../data/Dialect6/Multi_data/dev/pre_clean'
#
#train_file = '../data/Dialect6/Multistopwords/train'
#    
#test_file = '../data/Dialect6/Multistopwords/dev'

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


##word_vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,1))
#char_vect  = TfidfVectorizer(max_features = 50000, sublinear_tf=True,norm ='l1', max_df=0.75,analyzer = 'char_wb', ngram_range=(2,5))
#
#
#union = FeatureUnion([("w_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,2)
#                                 )),
#                       ("c_wb", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(2,5)
#                                 )),
##                       ("c_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(2,5)
##                                 ))
#                       ])
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

#print("%d documents - %0.3fMB (training set)" % (
#    len(data_train.data), data_train_size_mb))
#print("%d documents - %0.3fMB (test set)" % (
#    len(data_test.data), data_test_size_mb))
#print("%d categories" % len(target_names))
#print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target
l_acc = []
print("Extracting features from the training data using a sparse vectorizer")
#t0 = time()
#opts.use_hashing = True
#6
#union = FeatureUnion([
#        ("w_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,3)
#                                 )),
##        ("w_v2", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(2,3)
##                                 )),
#                       ("c_wb", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(2,6)
#                                 )),
##                       ("c_wb5", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(2,4)
##                                 )),
##                       ("c_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(5,5)
##                                 ))
#      ("sk",TfidfVectorizer(sublinear_tf=True, max_df=0.5,tokenizer=make_skip_tokenize(n=2, k=2)))
#
#                       ],
#transformer_weights={
#            'w_v': 0.7,
#            'c_wb': 0.5,
#           #' c_wb5':0.5,
#            'sk': 0.4,
#        }
#,
#)

union = FeatureUnion([
        ("w_v1", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,1)
                                 )),
        ("w_v2", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(2,2)
                                 )),
        ("w_v3", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(3,3)
                                 )),
        ("w_v4", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(4,4)
                                 )),
        ("w_v5", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(5,5)
                                 )),
                       ("c_wb1", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(2,2)
                                 )),
                        ("c_wb2", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(3,3)
                                 )),
                         ("c_wb3", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(4,4)
                                 )),
                          ("c_wb4", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(5,5)
                                 )),
#                           ("c_wb5", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(6,6)
#                                 )),
#                       ("c_wb5", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(2,4)
#                                 )),
#                       ("c_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(5,5)
#                                 ))
      ("sk",TfidfVectorizer(sublinear_tf=True, max_df=0.5,tokenizer=make_skip_tokenize(n=2, k=2)))

                       ],
transformer_weights={
            'w_v1': 0.7,
            'w_v2': 0.7,
            'w_v3': 0.7,
            'w_v4': 0.7,
            'w_v5': 0.7,
            'c_wb2': 0.6,
            'c_wb3': 0.6,
            'c_wb4': 0.6,
#            'c_wb5': 0.5,
            'c_wb1': 0.6,
           #' c_wb5':0.5,
            'sk': 0.4,
        }
,
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
conf_mat = metrics.confusion_matrix(y_test, pred)
print(conf_mat)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=['MSA', 'BEI', 'DOH', 'RAB', 'CAI', 'TUN'], yticklabels=['MSA', 'BEI', 'DOH', 'RAB', 'CAI', 'TUN'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()