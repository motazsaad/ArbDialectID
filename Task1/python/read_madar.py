#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:18:20 2019

@author: xabuka
"""

import os 
import cleaner

data_dir = '../../MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/'
train_file_6 = 'MADAR-Corpus-6-train.tsv'
dev_file_6 = 'MADAR-Corpus-6-dev.tsv'
train_file_26 = 'MADAR-Corpus-26-train.tsv'
dev_file_26 = 'MADAR-Corpus-26-dev.tsv'
#import csv


madar_list =[]
sents, labels= [],[]
# Firstly read tsv file
folder_path = data_dir+'Dialect26/train_set/'

def read_madar(file_name,tag):
    sent, label= [],[]
    with open(file_name, newline = '') as tsv:                                                                                         
        madar_list=[x.strip().split('\t') for x in tsv]
    for row in madar_list:
        sent.append(row[0]) # save sentence
        label.append(row[1]) # related dialect 
        if tag:
            store_set(row,folder_path)
        
        return sent, label
        # store a file inside the dialects folder ----- complete_dataset one_file/dialect
    #    with open(data_dir+'Dialect6/Dev_set/'+row[1]+'/'+row[1]+'.txt','a+') as write_file:
    #        write_file.write(row[0]+'\n') 



def store_set(row,folder_path):
    with open(folder_path+row[1]+'/'+row[1]+'.txt','a+') as write_file:
        write_file.write(row[0]+'\n') 

#clean_data
def save_cleaned_set(folder_path,labels):
    for file_name in set(labels):
        clean_data = cleaner.clean(folder_path+file_name+'/'+file_name+'.txt')
        print(len(clean_data))
        with open(folder_path+file_name+'/'+file_name+'_clean.txt','w+') as clean_file:
            for x in clean_data:
                clean_file.write(x+'\n')
        
        
def creat_splited_files(folder_path_to_read, folder_path_to_write, labels):
    #save every line in a file
    for lang in set(labels):
        with open(folder_path_to_read+lang+'/'+lang+'_clean.txt','r') as read_file:
            for i, line in enumerate(read_file.readlines()):
                #print(line)
                with open(folder_path_to_write+lang+'/'+lang+'_'+str(i)+'.txt','w+') as write_file:
                    write_file.write(line)



def create_folders(folder_path,labels):
    # make dilaects folders
    # folder for every dialect
    for i in set(labels):
        if not os.path.exists(folder_path+i):
            os.makedirs(folder_path+i)

#if not os.path.exists(data_dir+'Dialect6/Developing/pre_clean/'+i):

def remove_system_files(folder_path):
    #remove DS_store files
    for lang in set(labels):
        if os.path.exists(folder_path+lang+'/*.DS_store'):#+'/'+lang+'.txt'):
            print('yes')
            os.remove(folder_path+lang+'/*.DS_store')#+lang+'.txt')




if __name__ == '__main__':
    sents, labels= read_madar(data_dir+train_file_26,False)
    

