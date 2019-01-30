#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:18:20 2019

@author: xabuka
"""

#import csv
import os 

madar_list =[]
sent, labels= [],[]
with open('MADAR_TRAINING_SAMPLE.tsv', newline = '') as tsv:                                                                                          
    	madar_list=[x.strip().split('\t') for x in tsv]
       
for row in madar_list:
    sent.append(row[0])
    labels.append(row[1])
    # store a file inside the dialects folder
#    with open('dialects/'+row[1]+'/'+row[1]+'.txt','a+') as write_file:
#        write_file.write(row[0]+'\n')


# make dilaect folders
#for i in labels:
#    if not os.path.exists('dialects/'+i):
#        os.makedirs('dialects/'+i)


#print(sent)
#print(labels)

#split into single files
print(len(labels))
print(set(labels))


#save every line in a file
#for lang in set(labels):
#    with open('dialects/'+lang+'/'+lang+'.txt','r') as read_file:
#        for i, line in enumerate(read_file.readlines()):
#            print(line)
#            with open('dialects/'+lang+'/'+lang+'_'+str(i)+'.txt','w+') as write_file:
#                write_file.write(line)


#remove files
for lang in set(labels):
    if os.path.exists('single_data_set/dialects/'+lang+'/*.DS_store'):#+'/'+lang+'.txt'):
        print('yes')
        os.remove('single_data_set/dialects/'+lang+'/*.DS_store')#+lang+'.txt')
