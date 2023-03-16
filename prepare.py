#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, cast
from bs4 import BeautifulSoup
import os
from sklearn.model_selection import train_test_split
import requests
import nltk
import unicodedata
import re

def prepare_bills_for_processing(df):
    """
    Combines single line code and functions to prepare the data.
    """
    #Removes a bill with no text
    df = df[df.bill_text != "None"]
    df.bill_text = df.bill_text.apply(bill_trimmer)
    
    # creating a lemmatized column and cleaning the df
    df['lem']= df.bill_text.apply(clean_text)
    df['model']= df.lem.apply(join)
    return df
    
def bill_trimmer(input_string):
    """
    This function looks at bills and removes everything above 'A BILL' or 'RESOLUTION' or 'AN ACT'.
    """
    #Identifies the position of 'A BILL'
    text_pos = re.split('(A BILL|RESOLUTION|AN ACT)', input_string, 1)
    output_string = text_pos[2]
    return output_string


def clean_text(text, extra_stopwords=[]):

    '''
    This function takes in the words and cleans it, and returns the words that have been 
    lemmatized.
    '''

    # creating the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # adding an option to input stopwords
    stopwords = nltk.corpus.stopwords.words('english') + extra_stopwords
    
    # cleaning the text and making the text lower case, eliminating \n 
    # and anything that is not aplhanumeric
    clean_text = (unicodedata.normalize('NFKD', text)
                   .encode('ascii', 'ignore')
                   .decode('utf-8', 'ignore')
                   .lower())
    words = re.sub(r'[\n]', '', clean_text)
    words = re.sub(r'[^\w\s___]', '', clean_text).split()
    words = re.sub(r'_', '',' '.join(words)).split(' ')
    words = [w for w in words if len(w)<25]
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


def join(col):
    
    '''A function that joins the lemmatized and model columns'''
    
    return ' '.join(col)

def split_data(df, target):
    
    '''
    Splits a df into a train, validate, and test set. 
    target is the feature you will predict
    '''
    full = df
    train_validate, test = train_test_split(df, train_size =.8, random_state = 21)
    train, validate = train_test_split(train_validate, train_size = .7, random_state = 21)
    X_train = train.drop(columns=target)
    y_train = train[target]
    X_val = validate.drop(columns=target)
    y_val = validate[target]
    X_test = test.drop(columns=target)
    y_test = test[target]
    
    
    return train, X_train, y_train, X_val, y_val, X_test, y_test