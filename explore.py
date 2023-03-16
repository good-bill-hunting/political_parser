#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from env import api_key
import requests
import json
from acquire import *
from bs4 import BeautifulSoup
import os
from prepare import *
import matplotlib.pyplot as plt


def word_freq_new_df(df, clean_text):
    '''
    This function takes in a dataframe and the clean_text function
    to produce a new dataframe of words and word frequency rates.
    '''
    # creating a list of words for bill text and political parties
    all_bill_words = clean_text(' '.join(df['bill_text']))
    democrat_words = clean_text(' '.join(df[df['party'] == 'D']['bill_text']))
    republican_words = clean_text(' '.join(df[df['party'] == 'R']['bill_text']))
    ind_words = clean_text(' '.join(df[df['party'] == 'I']['bill_text']))
    
    # The value counts for the bill_text and political party words
    bill_counts = pd.Series(all_bill_words).value_counts()
    democrat_counts = pd.Series(democrat_words).value_counts()
    republican_counts = pd.Series(republican_words).value_counts()
    ind_counts = pd.Series(ind_words).value_counts()
    
    # concatinating the bill_text and political parties into one dataframe
    word_freq = pd.concat([bill_counts, democrat_counts, republican_counts, ind_counts], axis=1)
    word_freq.columns = ['bills', 'demo', 'repub', 'ind']
    
    # eliminating the most and least frequent words to reveal a more accurate depiction of specific political
    # parties and their respective top words
    word_freq = word_freq.loc[word_freq['bills'] <= 10000]
    word_freq = word_freq.loc[word_freq['bills'] >= 25]
    
    # filling the nan values with zero and making the df columns integers versus floats
    word_freq.fillna(0, inplace=True)
    word_freq = word_freq.astype('int')
    return word_freq


def demo_vis(df):
    '''
    This function creates a bar plot of the most common words appearing for democrats.
    '''
    
    #Plot the most frequent democratic words and color by label
    ax = df.sort_values('demo', ascending=False).head(5).plot.bar(color=['lightgrey', 'royalblue', 'red', 'green'], figsize=(16, 9))
    plt.title('Most Common Words for Democrats')
    plt.ylabel('Count')
    plt.xlabel('Most Common Words')
    plt.xticks(rotation=45)
    ax.legend(['Bills', 'Democrat', 'Republican', 'Independent'])
    return plt.show()


def repub_vis(df):
    '''
    This function creates a bar plot of the most common words appearing for republicans.
    '''
    
    #Plot the most frequent democratic words and color by label
    ax = df.sort_values('repub', ascending=False).head(5).plot.bar(color=['lightgrey', 'royalblue', 'red', 'green'], figsize=(16, 9))
    plt.title('Most Common Words for Republicans')
    plt.ylabel('Count')
    plt.xlabel('Most Common Words')
    plt.xticks(rotation=45)
    ax.legend(['Bills', 'Democrat', 'Republican', 'Independent'])
    return plt.show()



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



def ind_vis(df):
    '''
    This function creates a bar plot of the most common words appearing for democrats.
    '''
    
    #Plot the most frequent independent words and color by label
    ax = df.sort_values('ind', ascending=False).head(5).plot.bar(color=['lightgrey', 'royalblue', 'red', 'green'], figsize=(16, 9))
    plt.title('Most Common Words for Independents')
    plt.ylabel('Count')
    plt.xlabel('Most Common Words')
    plt.xticks(rotation=45)
    ax.legend(['Bills', 'Democrat', 'Republican', 'Independent'])
    return plt.show()


def demo_trigrams_vis(df):

    more_stopwords = ['secretary','united','states','senate','house','representative',
                   'representatives','fiscal','year','shall','adding','end','paragraph',
                   'made','available','prebody','subsection','day','date','submit','described',
                   'may','congress','following','new','enactment','code','section','assembled',
                   'b','amended','short','title','sec','heading', 'et', 'seq',
                    'chapter', 'effective','enacted','subchapter','entity', '42', 'usc', 'act', 'establish', 'categorical', 'america', '1', '2', 'seq']

    demo_words = clean_text(' '.join(df[df['party'] == 'D']['bill_text']), more_stopwords)
    
    demo_trigrams = pd.Series(nltk.ngrams(demo_words, 3))
    
    top_demo_trigrams = demo_trigrams.value_counts().head(10)
    
    #Plot democrat trigrams
    top_demo_trigrams.plot.barh().invert_yaxis()
    plt.xlabel('Count')
    plt.ylabel('Trigrams')
    plt.title('Commonly occurring democrat trigrams')
    return plt.show()



def repub_trigrams_vis(df):

    more_stopwords = ['secretary','united','states','senate','house','representative',
                   'representatives','fiscal','year','shall','adding','end','paragraph',
                   'made','available','prebody','subsection','day','date','submit','described',
                   'may','congress','following','new','enactment','code','section','assembled',
                   'b','amended','short','title','sec','heading', 'et', 'seq',
                    'chapter', 'effective','enacted','subchapter','entity', '42', 'usc', 'act', 'establish', 'categorical', 'america', '1', '2', 'seq']
    
    repub_words = clean_text(' '.join(df[df['party'] == 'R']['bill_text']), more_stopwords)
    
    repub_trigrams = pd.Series(nltk.ngrams(repub_words, 3))
    
    top_repub_trigrams = repub_trigrams.value_counts().head(10)
    
    #Plot republican trigrams
    top_repub_trigrams.plot.barh().invert_yaxis()
    plt.xlabel('Count')
    plt.ylabel('Trigrams')
    plt.title('Commonly occurring republican trigrams')
    return plt.show()



def ind_trigrams_vis(df):

    more_stopwords = ['secretary','united','states','senate','house','representative',
                   'representatives','fiscal','year','shall','adding','end','paragraph',
                   'made','available','prebody','subsection','day','date','submit','described',
                   'may','congress','following','new','enactment','code','section','assembled',
                   'b','amended','short','title','sec','heading', 'et', 'seq',
                    'chapter', 'effective','enacted','subchapter','entity', '42', 'usc', 'act', 'establish', 'categorical', 'america', '1', '2', 'seq']
    
    ind_words = clean_text(' '.join(df[df['party'] == 'I']['bill_text']), more_stopwords)
    
    ind_trigrams = pd.Series(nltk.ngrams(ind_words, 3))
    
    top_ind_trigrams = ind_trigrams.value_counts().head(10)
    
    top_ind_trigrams.plot.barh().invert_yaxis()
    plt.xlabel('Count')
    plt.ylabel('Trigrams')
    plt.title('Commonly occurring independent trigrams')
    return plt.show()


# In[ ]:




