#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:19:29 2019

@author: xabuka
"""
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import os
import cleaner


#maxlen = 100  # We will cut reviews after 100 words
#training_samples = 700  # We will be training on 200 samples
#validation_samples = 200  # We will be validating on 10000 samples
#max_words = 10000  # We will only consider the top 10,000 words in the dataset
#imdb_dir = '../splitedPalSent'




def load_train(imdb_dir,maxlen,training_samples, validation_samples,max_words, Validation = True, binary= False ):
#train_directory
    
    train_dir = os.path.join(imdb_dir, 'train')
    print(train_dir)
    labels = []
    texts = []
    #i = 0
    target_class = ['MSA', 'BEI', 'DOH', 'RAB', 'CAI', 'TUN']
    #if not binary:
     #   target_class.append('NO') 
    
    for label_type in target_class:#categories
#    for label_type in ['pos','neg']: # for mdb
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] != 'tore':
                f = open(os.path.join(dir_name, fname))
                #i += 1
                #print(fname) 
                
                texts.append(f.read())# add the sentences to text array
                f.close()
                if label_type in ['MSA']:# which value to assign to every class
                    labels.append(0)
                elif label_type in ['BEI']:# which value to assign to every class
                    labels.append(1)
                elif label_type in ['DOH']:# which value to assign to every class
                    labels.append(2)
                elif label_type in ['RAB']:# which value to assign to every class
                    labels.append(3)
                elif label_type in ['CAI']:# which value to assign to every class
                    labels.append(4)
                elif label_type in ['TUN']:# which value to assign to every class
                    labels.append(5)
                
            
    
#    for sent,clas in zip(texts,labels):
#        print(sent,clas)
    
    #datatrain_texts = texts
    #print(len(datatrain_texts))
    tokenizer = Tokenizer(num_words=max_words,split=' ')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
#    print(len(sequences))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen,padding='post')
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    # Split the data into a training set and a validation set
    # But first, shuffle the data, since we started from data
    # where sample are ordered (all negative first, then all positive).
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    if not Validation:
        training_samples= len(data)
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    if not binary:
        y_train = to_categorical(y_train, num_classes=len(target_class))


    
    if Validation: 
        x_val = data[training_samples: training_samples + validation_samples]
        y_val = labels[training_samples: training_samples + validation_samples]
        if not binary:
            y_val = to_categorical(y_val, num_classes=len(target_class))
        print(len(x_train))
        print(len(x_val))
        

        return x_train, y_train, x_val, y_val
    
        
    else:
        #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        #return x_train, x_test, y_train, y_test
        return x_train, y_train
    
    

    




def load_test(imdb_dir,maxlen,max_words,binary= False):
    
    test_dir = os.path.join(imdb_dir, 'dev')
    labels = []
    texts = []
    target_class = ['MSA', 'BEI', 'DOH', 'RAB', 'CAI', 'TUN']
    #if not binary:
     #   target_class.append('NO') 
    
    for label_type in target_class:#categories
#    for label_type in ['pos','neg']: # for mdb

        dir_name = os.path.join(test_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] != 'tore':
                f = open(os.path.join(dir_name, fname))
                #i += 1
                #print(fname) 
                texts.append(f.read())# add the sentences to text array
                f.close()
                if label_type in ['MSA']:# which value to assign to every class
                    labels.append(0)
                elif label_type in ['BEI']:# which value to assign to every class
                    labels.append(1)
                elif label_type in ['DOH']:# which value to assign to every class
                    labels.append(2)
                elif label_type in ['RAB']:# which value to assign to every class
                    labels.append(3)
                elif label_type in ['CAI']:# which value to assign to every class
                    labels.append(4)
                elif label_type in ['TUN']:# which value to assign to every class
                    labels.append(5)
             
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x_test = pad_sequences(sequences, maxlen=maxlen,padding='post')
    
    #print(labels)
    y_test = np.asarray(labels)
    
    if not binary:
        y_test = to_categorical(y_test, num_classes=len(target_class))

    
    #print(y_test)
    return x_test, y_test



# we need it for word_embedding Ara-vec, later I will upadte the code


def word_index(imdb_dir,maxlen,max_words):
    train_dir = os.path.join(imdb_dir, 'train')
    labels = []
    texts = []
    target_class = ['MSA', 'BEI', 'DOH', 'RAB', 'CAI', 'TUN']
#    for label_type in ['pos','neg']: # for mdb
    for label_type in target_class:#categories

        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] != 'tore':
                f = open(os.path.join(dir_name, fname))
                #i += 1
                #print(fname) 
                texts.append(f.read())# add the sentences to text array
                f.close()
                if label_type in ['MSA']:# which value to assign to every class
                    labels.append(0)
                elif label_type in ['BEI']:# which value to assign to every class
                    labels.append(1)
                elif label_type in ['DOH']:# which value to assign to every class
                    labels.append(2)
                elif label_type in ['RAB']:# which value to assign to every class
                    labels.append(3)
                elif label_type in ['CAI']:# which value to assign to every class
                    labels.append(4)
                elif label_type in ['TUN']:# which value to assign to every class
                    labels.append(5)
            
    
    
    #datatrain_texts = texts
    #print(len(datatrain_texts))
    tokenizer = Tokenizer(num_words=max_words,split=' ')
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    
    return word_index



def one_encode_pred(yhat):
    yhat_enoced = []
    
    for x in yhat:
        new_a = [0,0,0]
        old_a = list(x)
        maxpos = old_a.index(max(old_a))
        new_a[maxpos]= 1
        yhat_enoced.append(new_a)
#        print(x)
#        print(new_a)
    return yhat_enoced
        


