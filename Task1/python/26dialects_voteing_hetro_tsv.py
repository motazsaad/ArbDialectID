#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:08:25 2019

@author: xabuka
"""

# best Identify langauge code


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import FunctionTransformer
import pandas as pd 




def read_datafram(file_name):
    data = pd.read_csv(file_name,';',names = ["Class", "6D", "Body","len1","len"])
    data = data.drop(columns="len1")
    data = pd.DataFrame(data)
    return data






Test = True
data_train = read_datafram('result26/train3.csv')

 
if Test :
    data_test = read_datafram('result26/test3.csv')
else:
    data_test = read_datafram('result26/dev3.csv')

    


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

estimators = []
#sgd = SGDClassifier(alpha=0.00001, max_iter=50,penalty="l2") 
#estimators.append(('sgd', sgd))
svc = LinearSVC(penalty='l2', dual=False,tol=1e-3)
estimators.append(('svc',svc))
mnb= MultinomialNB(alpha=.01)
estimators.append(('mnb',mnb))
bnb= BernoulliNB(alpha=.01)
estimators.append(('bnb',bnb))

ensemble = VotingClassifier(estimators)

# split a training set and a test set
print("Extracting features from the training data using a sparse vectorizer")
#t0 = time()
#opts.use_hashing = True
#6
max_df = 0.5
##min_df = 1
union = FeatureUnion([("w_v", TfidfVectorizer(sublinear_tf=True, max_df=max_df,analyzer = 'word', ngram_range=(1,3)
                                 )),
#        ("w_v2", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(2,3)
#                                 )),
                       ("c_wb", TfidfVectorizer(sublinear_tf=True,max_df=max_df,analyzer = 'char_wb', ngram_range=(2,5)
                                 )),
                       ("c_wb5", TfidfVectorizer(sublinear_tf=True, max_df=max_df,analyzer = 'char', ngram_range=(2,4)
                                 )),
#                       ("c_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(5,5)
#                                 ))
      ("sk",TfidfVectorizer(sublinear_tf=True, max_df=max_df,tokenizer=make_skip_tokenize(n=2, k=1)))

                       ],
transformer_weights={
            'w_v': 0.5,
            'c_wb': 0.5,
           ' c_wb5':0.5,
            'sk': 0.3,
        }
,
)
    
    
from sklearn.base import BaseEstimator,TransformerMixin
class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()


get_body_data = FunctionTransformer(lambda x: x['Body'], validate=False)
get_6d = FunctionTransformer(lambda x: x[['6D']], validate=False)

get_len = FunctionTransformer(lambda x: x[['len']], validate=False)



process_and_join_features = Pipeline([
    ('features', FeatureUnion([
            
    
    ('6D_featuresg', Pipeline([
                ('selector', get_6d)
                
            ])),
    ('len_featuresg', Pipeline([
                ('selector', get_len)
                
            ])),
             ('text_features', Pipeline([
                ('selector', get_body_data),
                ('un',union),

            ]))
         ]
#            ,
#            transformer_weights={
#            
#            '6D_featuresg': 0.7,
#           'text_features': 1,
#
#       }
)),
    ('clf',ensemble#MultinomialNB(alpha=.01)
    )
])

model = process_and_join_features.fit(data_train,data_train['Class'])
print("Combined space has", data_train.shape[1], "features")
print(data_train.shape)
print(data_test.shape)
pred = model.predict(data_test)

if  Test:
    score = metrics.accuracy_score(data_test['Class'], pred)
    print("accuracy:   %0.3f" % score)   
    
    print("classification report:")
    print(metrics.classification_report(data_test['Class'], pred,target_names=set(data_test['Class'])))   
        






if Test:
    file_type = 'hetro_test_paper'
else:
    file_type = 'hetro_dev_paper'

#gold_test_file =  open('result26/Final_'+file_type+'_gold_100.txt','w+') 
#pred_test_file = open('result26/Final_'+file_type+'_pred_100.txt','w+')
#sample_file = open('result26/Final_'+file_type+'_set_100.txt','w+')
##
##Final_hetro_dev_set_100 
#counter = 0 
#if not Test:  
#    for target,pre,doc in zip(data_test['Class'],pred,data_test['Body']):
#        counter += 1
#        gold_test_file.write(target+ '\n')
#        pred_test_file.write(pre+ '\n')
#        sample_file.write(target+'\t'+pre+'\t'+' '+doc+'\n')
#else:
#    for pre,doc in zip(pred,data_test['Body']):
#        counter += 1
#        pred_test_file.write(pre+ '\n')
#        sample_file.write(pre+'\t'+' '+doc+'\n')
#
#print(counter)  
#gold_test_file.close() 
#pred_test_file.close()
#sample_file.close() 



