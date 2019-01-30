
from langid.langid import LanguageIdentifier, model

import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--number_of_grams", "-n", type=str, help='enter number of grams', required=True)
parser.add_argument("--test_file_name", "-f", type=str, help='enter file_name', required=True)

args = parser.parse_args()
with open (args.test_file_name,'r') as file_reader:
    docs = list(file_reader.readlines())
#docs = ["ولك من وين انت جاي ياخو ينعن هيك عيشة معبر رفح؟","كيف حالك  شو اخبارك انا منيح","دخيل قلبك شو مهضوم تؤبر قلبي","ماعاد فينا نتحمل البسينة عم تغط عنفسي"]



arabic_dialect_model = open('third_MSA_model_'+args.number_of_grams+'_grams/model').read()#open('built_models/exp2/corpus_model_4').read()
identifier = LanguageIdentifier.from_modelstring(arabic_dialect_model, norm_probs=True)
total_prob = 0
total =[0]*(len(identifier.nb_classes)+1)#[0,0,0,0,0,0,0,0,0]
#print(len(identifier.nb_classes))
print('#################################')
print('test '+args.number_of_grams+'-grams model')
for ii,doc in enumerate(docs) :
    print(doc, identifier.classify(doc)) #jo
    fv = identifier.instance2fv(doc)
    probs = identifier.norm_probs(identifier.nb_classprobs(fv))
    # print(probs)
    # print(identifier.nb_classes)
    #print every class and probability
    # for pro, clas in zip (probs, identifier.nb_classes):
    #     print(pro,clas)
    for i, prob in enumerate(probs):
        total[i] += probs[i]
    	
        
    #print('probabilty for MSA Class')
    #total_prob = total_prob+ probs[5]
for t in range(2,len(total)-1):
    print('class',str(identifier.nb_classes[t]),'=',total[t]/len(docs))#+' = '+ str(total[t]/len(docs))
    #print('=',total[t]/len(docs))

#print('number of lines = ', len(docs))
#print('total prob = ',str(total_prob/len(docs)))


