import glob
import os
import eval_metrics
from langid.langid import LanguageIdentifier
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--number_of_grams", "-n", type=str, help='enter number of grams', required=True)

args = parser.parse_args()
#n = '2'
#model location 
#/Users/xabuka/PycharmProjects/measuring_acceptability/corpora/models
arabic_dialect_model = open('corpora/models/nizar/dialects_model_'+args.number_of_grams+'_grams/model').read()
identifier = LanguageIdentifier.from_modelstring(arabic_dialect_model, norm_probs=True)

test_dir = 'corpora/parallel_corpus/test_padic/one__file/*.test'
test_files = glob.glob(test_dir)

for test in test_files:
    base_name, ext = os.path.basename(test).split('.')
    lines = open(test).read().splitlines()
    prediction_list = list()
    y_list = list()
    for line in lines:
        if line:
            dialect, conf = identifier.classify(line)
            prediction_list.append(dialect)
            y_list.append(base_name)
            #print(dialect, base_name)
    print('evaluating ', base_name)
    accuracy, precision, recall, f_score = eval_metrics.evaluate_acc_PRF(prediction_list, y_list, base_name)
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f-score: {}'.format(f_score))
    print('-------------------------------------')
