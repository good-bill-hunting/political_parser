import re
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.snowball import EnglishStemmer

import pandas as pd
import numpy as np
import requests
import os
import json
import unicodedata
import nltk

from sklearn.model_selection import train_test_split

#Uncomment to download extra data for nltk
#nltk.download('omw-1.4')
#nltk.download("punkt")
#nltk.download("stopwords")
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')
#nltk.download("maxent_ne_chunker")
#nltk.download("words")
#nltk.download("book") #big download

#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")

def tokenized(input_string, tokenize_tool=1, return_list=False):
    """
    Input:
    This function takes in a string and tokenizer tool argument and returns a list of tokens.
    tokenize_tool=1: ToktokTokenizer
    tokenize_tool=2: word_tokenizer
    tokenize_tool=3: sent_tokenizer
    """
    
    if tokenize_tool==1:
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(input_string, return_str=True)
    elif tokenize_tool == 2:
        tokens = word_tokenize(input_string)
    elif tokenize_tool == 3:
        tokens = sent_tokenize(input_string)
    if return_list:
        return tokens
    else:
        token_string = ' '.join(tokens)
        return token_string   

def basic_clean(input_string):
    """
    This function takes in a string and applies basic cleaning to it.
    """
    #Changes all characters to their lower case.
    input_string = input_string.lower()

    input_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    #Removes special characters
    input_string = re.sub(r"[^a-z0-9\s]", ' ', input_string)
    input_string = re.sub(r'[\n]', ' ', input_string)


    return input_string

def remove_stopwords(input_string, extra_words=None, exclude_words=None, return_list=False):
    """
    This function takes an input and removes stopwords. You can add or remove words witht the extra_words or exclude_words args.
    """
    
    stopword_list = stopwords.words('english')
    if extra_words != None:
        stopword_list.extend(extra_words)
    if exclude_words != None:
        if type(extra_words) == str:
            stopword_list.remove(exclude_words)
        if type(extra_words) == list:
            for word in exclude_words:
                stopword_list.remove(word)
    words = input_string.split()
    filtered_words = [w for w in words if w not in stopword_list]
    if return_list:
        return filtered_words
    else:
        string_without_stopwords = ' '.join(filtered_words)
        return string_without_stopwords


def lemmatized(input_string, return_list=False):
    """
    Takes an input string and lemmatizes it.
    Please do not stem and lemmatize the same string.
    """
    #Creates the lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()
    
    #Makes lemmatade
    lemmas = [wnl.lemmatize(word) for word in input_string.split()]
    if return_list:
        return lemmas
    lemmatized_string = ' '.join(lemmas)
    
    return lemmatized_string

def stemmerize_tool(input_string, stemmer_type=1, return_list=False):
    """
    Input a string of words to stemmertize. Returns a string.
    stemmer_type=1 - PorterStemmer
    stemmer_type=2 - EnglishStemmer
    stemmer_type=3 - SnowballStemmer("english")
    """
    
    if stemmer_type ==1:
        stemmer = PorterStemmer()
    elif stemmer_type ==2:
        stemmer = EnglishStemmer()
    elif stemmer_type ==3:
        stemmer = SnowballStemmer("english")
    stem_list = [stemmer.stem(word) for word in input_string.split()]
    if return_list:
        return stem_list
    return ' '.join(stem_list)


def train_validate(df, stratify_col = None, random_seed=1969):
    """
    This function takes in a DataFrame and column name for the stratify argument (defualt is None).
    It will split the data into three parts for training, testing and validating.
    """
    #This is logic to set the stratify argument:
    stratify_arg = ''
    if stratify_col != None:
        stratify_arg = df[stratify_col]
    else:
        stratify_arg = None
    
    #This splits the DataFrame into 'train' and 'test':
    train, test = train_test_split(df, train_size=.8, stratify=stratify_arg, random_state = random_seed)
    
    #The length of the stratify column changed and needs to be adjusted:
    if stratify_col != None:
        stratify_arg = train[stratify_col]
        
    #This splits the larger 'train' DataFrame into a smaller 'train' and 'validate' DataFrames:
    train, validate = train_test_split(train, train_size=.6, stratify=stratify_arg, random_state = random_seed)
    return train, validate, test


def extract_proper_nouns(quote):
    """
    Intakes a string and returns words tagged as proper nouns.
    """
    words = word_tokenize(quote)
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(" ".join(i[0] for i in t)for t in tree if hasattr(t, "label") and t.label() == "NE")


def ngrams_creator(input_string, n_grams = 2):
    """
    This function takes in a list and returns a list of grams.
    """
    ngrams = nltk.ngrams(input_string.split(), n_grams)
    return list(ngrams)

def big_func_to_pre_data(df):
    """
    This function is a function of functions that prepare the data.
    """
    #removes null values in all columns
    df.dropna(inplace=True)
    
    #Performs functions listed above
    df = clean_languages(df)
    df['readme_clean'] = df['readme_contents'].apply(basic_clean)
    df['readme_clean'] = df['readme_clean'].apply(tokenized, tokenize_tool=2)
    df['readme_stem'] = df['readme_clean'].apply(stemmerize_tool, stemmer_type=3)
    
    return df

def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates a series with only the target variable to test accuracy.
    """
    #Seperating out the target variable
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_val = val.drop(columns = [target_col])
    y_val = val[target_col]

    X_test = test.drop(columns = [target_col])
    y_test = test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test