"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

#from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
op.add_option("--train_folder",
              action="store", type = "str",
              help="Enter the Train directory.")
op.add_option("--test_folder",
              action="store", type= "str",
              help="Enter the test directory.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


# #############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
            'jo','lb', 'pa' , 'sy',
#        'alt.atheism',
#        'talk.religion.misc',
#        'comp.graphics',
#        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading MADAR dataset for categories:")
print(categories if categories else "all")

#arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
#print(list(arb_stopwords))
#stop_words = frozenset(arb_stopwords)

if opts.train_folder:
    train_file = opts.train_folder
else:
    train_file = '../../MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/Dialect26/Training/pre_clean'
    
if opts.test_folder:
    test_file = opts.test_folder
else:
    test_file = '../../MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/Dialect26/Developing/pre_clean'
"""

4. Madar
../single_data_set/dialects

"""
data_train = load_files(train_file, encoding = 'utf-8',decode_error='ignore')

data_test = load_files(test_file, encoding = 'utf-8',decode_error='ignore')

print('data loaded')
#print(data_test.data)
# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names

#print(len(data_test.data))
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(target_names))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target


print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
#opts.use_hashing = True
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


def save_result_file(target_test,pred_test,name,X_samples):
    gold_test_file =  open(name+'_gold.txt','w+') 
    pred_test_file = open(name+'_pred.txt','w+')
    sample_file = open(name+'_test_set.txt','w+')
    
    for target,pred,sent in zip(target_test,pred_test,name,X_samples):
        gold_test_file.write(data_test.target_names[target])
        pred_test_file.write(data_test.target_names[pred])
        sample_file.write(vectorizer.inverse_transform(sent))
    
    
    
    
# #############################################################################
# Benchmark classifiers
def benchmark(clf,name):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
#    for doc,category in zip(X_test, pred):
#        print('%d => %s' % (category, data_test.target_names[category]))
#        print(' '.join(vectorizer.inverse_transform(doc)[0]))
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    #print(pred)
    #print(vectorizer.inverse_transform(X_test))

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    #opts.print_top10 = True
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    opts.print_report = True
    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    #opts.print_cm = True
    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    
    # save files
    gold_test_file =  open('result26/'+name+'_gold.txt','w+') 
    pred_test_file = open('result26/'+name+'_pred.txt','w+')
    sample_file = open('result26/'+name+'_test_set.txt','w+')
    
    for target,pre,doc in zip(y_test,pred,X_test):
        gold_test_file.write(data_test.target_names[target]+ '\n')
        pred_test_file.write(data_test.target_names[pre]+ '\n')
        sample_file.write(data_test.target_names[target]+'\t'+' '.join(vectorizer.inverse_transform(doc)[0])+'\n')
    
    
    
    
#    print('the test_part')
#    
#    docs_new = ['انا مش عارف شو الموقف حاليا ', 'كل صالونات لبنان تتحدث عن رائحه فساد ', 'بعرفش هيا هيك شو اعملك ']
#    x_doc = vectorizer.transform(docs_new)    
#    predicted = clf.predict(x_doc)
#    for doc,category in zip(docs_new, predicted):
#        print('%r =>%d => %s' % (doc,category, data_test.target_names[category]))
        
    return clf_descr, score, train_time, test_time
    



results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge_Classifier"),
        (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random_forest")
       ):
    print('=' * 80)
    print(name)# the name of the classfier
    results.append(benchmark(clf,name))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3),'LinearSVC_'+penalty))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty),'SGDClassifier_'+penalty))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet"),'SDG_Elastic-Net_penalty'))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid(),'NearestCentroid_aka_Rocchio_classifier'))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.1),'Naive_Bayes_MultinomialNB_alpha_0.1'))
results.append(benchmark(BernoulliNB(alpha=.1),'Naive_Bayes_BernoulliNB_alpha_0.1'))
results.append(benchmark(ComplementNB(alpha=.1),'Naive_Bayes_ComplementNB_alpha_0.1'))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))]),'LinearSVC_with_L1-based_feature_selection'))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

for i, c, s in zip(indices, clf_names,score):
    print(i,c,"%0.3f" % s)
   # print("%0.3f" % scores)