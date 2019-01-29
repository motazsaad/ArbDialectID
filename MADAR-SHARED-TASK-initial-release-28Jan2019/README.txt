===============================================================
                    MADAR SHARED TASK 2019
            Arabic Fine-Grained Dialect Identification

   The Fourth Workshop for Arabic Natural Language Processing
                         (WANLP 2019)

                    Initial Release of Data
                          28 Jan 2019
===============================================================

This README file is present in the zipped folder: 
MADAR-SHARED-TASK-initial-release-28Jan2019.zip
 
MADAR_TRAINING_SAMPLE.tsv contains a small sample of the full 
training set of the MADAR Shared Task (Subtask 1). It includes 
520 sentences along with their dialect labels.

There are 520 lines in the file, each line has two tab separated 
columns. The first column contains Arabic sentences, and the 
second column contains corresponding dialectal labels. 

There are 20 sentences translated into dialectal Arabic from 25 
cities of the Arabic Region and Modern Standard Arabic 
(A total of (25+1)*20 = 520).

The dialect city labels are defined in the following publication:

Salameh, Mohammad, Houda Bouamor, and Nizar Habash. "Fine-Grained 
Arabic Dialect Identification." Proceedings of the 27th International
Conference on Computational Linguistics. 2018.  
http://aclweb.org/anthology/C18-1113

MADAR-DID-Scorer.py is a python script that will take in two text 
files containing true labels and predicted labels and will output 
accuracy, F1 score, precision and recall. (Note that the final 
ranking will use F1 score).

Please make sure to have sklearn library installed.


Usage of the scorer:

    python MADAR-DID-Scorer.py  <gold-file> <pred-file>

    For verbose mode:
        
    python MADAR-DID-Scorer.py  <gold-file> <pred-file> -verbose

In the provided directory there are example gold and prediction files. 
If they are used with the scorer, they should produce the following 
results:


python3 MADAR-DID-Scorer-v.0.1.py EXAMPLE.GOLD EXAMPLE.PRED

OVERALL SCORES:
MACRO AVERAGE PRECISION SCORE: 80.56 %
MACRO AVERAGE RECALL SCORE: 75.00 %
MACRO AVERAGE F1 SCORE: 73.89 %
OVERALL ACCURACY: 75.00 %


===============================================================
Copyright (c) 2019 Carnegie Mellon University Qatar
and New York University Abu Dhabi. All rights reserved.
===============================================================

