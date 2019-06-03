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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection


#gold_test_file =  open('result26/Final_dev_gold_100.txt','w+') 
#pred_test_file = open('result26/Final_dev_pred_100.txt','w+')
#sample_file = open('result26/Final_dev_set_100.txt','w+')
## 
#counter = 0   
#for target,pre,doc in zip(y_test,pred,data_test):
#    counter += 1
#    gold_test_file.write(target+ '\n')
#    pred_test_file.write(pre+ '\n')
#    sample_file.write(target+'\t'+' '+doc+'\n')
#
#print(counter)


import csv
import numpy
#train by 6 dialects
filename = '../data/MADAR-Corpus-6-train.tsv'

raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter='\t', quoting=csv.QUOTE_NONE)
x = list(reader)
filename = '../data/MADAR-Corpus-6-dev.tsv'
reader = csv.reader(raw_data, delimiter='\t', quoting=csv.QUOTE_NONE)
x = x + list(reader)
data = numpy.array(x)#.astype('string')
print(data.shape)
data_train = data[:,0]
y_train = data[:,1]

#test by 26 dialect
#filename = '../data/MADAR-Corpus-26-train.tsv'
#raw_data = open(filename, 'rt')
#reader = csv.reader(raw_data, delimiter='\t', quoting=csv.QUOTE_NONE)
#x = list(reader)
#data = numpy.array(x)#.astype('string')
#print(data.shape)
#data_train = data[:,0]
#y_train = data[:,1]



filename = '../data/MADAR-Corpus-26-test2.tsv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter='\t', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x)#.astype('string')
print(data.shape)
data_test = data[:,0]
y_test = data[:,1]


#filename = '../data/MADAR-Corpus-26-test.tsv'
#raw_data = open(filename, 'rt')
#reader = csv.reader(raw_data, delimiter='\t', quoting=csv.QUOTE_NONE)
#x = list(reader)
#data = numpy.array(x)#.astype('string')
#print(data.shape)
#data_test = data[:,0]



print("Loading MADAR dataset for categories:")




print("Traing Data:   {0}".format(len(data_train)))
print("Testing Data:   {0}".format(len(data_test)))


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



# split a training set and a test set
l_acc = []
print("Extracting features from the training data using a sparse vectorizer")
#t0 = time()
#opts.use_hashing = True
#6
max_df = 0.5

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
X_train = union.fit_transform(data_train) #union.fit_transform(data_train.data)
X_test = union.transform(data_test)

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


pred = ensemble.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, pred,target_names=set(y_test)))

#print("confusion matrix:")
#print(metrics.confusion_matrix(y_test, pred))

print('length')
print(len(y_test))
print(len(pred))
print(len(data_test))
#
#   #save files
file_type = ['6-26dev','6-26test2']
gold_test_file =  open('result26/Final_'+file_type[1]+'_gold_100.txt','w+') 
pred_test_file = open('result26/Final_'+file_type[1]+'_pred_100.txt','w+')
sample_file = open('result26/Final_'+file_type[1]+'_set_100.tsv','w+')
# 
counter = 0   
for target,pre,doc in zip(y_test,pred,data_test):
    counter += 1
    gold_test_file.write(target+ '\n')
    pred_test_file.write(pre+ '\n')
    sample_file.write(target+'\t'+pre+'\t'+' '+doc+'\n')

print(counter)  
gold_test_file.close() 
pred_test_file.close()
sample_file.close() 
#for pre,doc in zip(pred,data_test):
#    pred_test_file.write(pre+ '\n')
#    sample_file.write(doc+'\n')
    

    
    


