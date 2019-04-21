#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:05:30 2018

@author: xabuka
"""

import sys
import string
import alphabet_detector
from pprint import pprint
import delete_repeated_char as del_char
import re




RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)







emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def tokenize(s):
    return tokens_re.findall(s)


def normalize_arabic(text):
    text = remove_diacritics(text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text




def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)

def remove_punctuations(text):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؟'
    return ''.join(ch for ch in text if ch not in punctuations)


def remove_punctuation(s):
    translator = str.maketrans('', '', string.punctuation + "،")
    return s.translate(translator)


def keep_only_arabic(words):
    ad = alphabet_detector.AlphabetDetector()
    tokens = [token for token in words if ad.is_arabic(token)]
    tweet = ' '.join(tokens)
    return tweet


def remove_digits(text):
    remove_digits = str.maketrans('', '', string.digits)
    res = text.translate(remove_digits)
    return res


def clean(corpus_file):
    
    print("reading corpus ...")
    corpus = open(corpus_file).readlines()
    new_corpus = []
    #pprint(corpus)
    
    for line in corpus:
        
    
        #print("removing punctuations and digits")
        clean_text = remove_punctuations(remove_punctuation(line))
        #norm_text  = normalize_arabic(clean_text)
        #clean_text = remove_diacritics(clean_text)
        alphapet_text = remove_digits(clean_text)
        #del corpus
        #del clean_text
        #del norm_text
    
        #print("remove non Arabic")
        pure_arabic_text = keep_only_arabic(alphapet_text.split())
        #del alphapet_text

        final_update_text = del_char.delete_repeat_char(pure_arabic_text)

        final_without_emoji = strip_emoji(final_update_text)
        new_corpus.append(pure_arabic_text)
        #new_corpus.append(final_without_emoji)
    
    #print(new_corpus)
    #return ' '.join(words), new_corpus
    return new_corpus


#print as one line
def print_to_file(new_corpus,output_file):
    fout = open(output_file, 'w', encoding = 'utf-8')
    for line in new_corpus:
        if not line.strip(): continue # remove empty lines
        fout.write(line+'\n') 

def usage():
    return "please provide a corpus file"


if __name__ == '__main__':
    if len(sys.argv) == 3:
        folder_name = 'clean_data/' +sys.argv[1]
        corpus = folder_name + '/' +sys.argv[2]  # file_name
        #print(clean(corpus))
        print_to_file(clean(corpus),folder_name+'/clean_'+sys.argv[2])
    else:
        print_to_file(clean('1.txt'),'11.txt')
        print(usage())
        sys.exit(-1)

