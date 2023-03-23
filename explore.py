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

def republican_trigram_viz(df):
    more_stopwords = ['secretary','united','states','senate','house','representative',
                   'representatives','fiscal','year','shall','adding','end','paragraph',
                   'made','available','prebody','subsection','day','date','submit','described',
                   'may','congress','following','new','enactment','code','section','assembled',
                   'b','c','amended','short','title','sec','heading', 'et', 'seq',
                    'chapter', 'effective','enacted','subchapter','entity', '42', 'usc', 'act', 'establish',
                       'categorical', 'america', '1', '2', 'seq','authorization',
                       'appropriations', 'appropriated', 'inserting','numerical',
                       'sequence','ii']
    
    democrat_words = clean_text(' '.join(df[df['party'] == 'D']['bill_text']), 
                                 more_stopwords)
    
    republican_trigrams = pd.Series(nltk.ngrams(republican_words, 3))
    top_republican_trigrams =pd.DataFrame(republican_trigrams.value_counts().head(40))
    democrat_trigrams = pd.Series(nltk.ngrams(democrat_words, 3))
    top_democrat_trigrams =pd.DataFrame(democrat_trigrams.value_counts().head(40))
    top_democrat_trigrams['party'] = 'D'
    top_republican_trigrams['party'] = 'R'
    top_democrat_trigrams.reset_index(inplace = True)
    top_republican_trigrams.reset_index(inplace = True)
    top_democrat_trigrams.rename(columns={"index": "trigram", 
                                       0: "frequency"},inplace = True)

    top_republican_trigrams.rename(columns={"index": "trigram", 
                                       0: "frequency"},inplace = True)
    scaler = StandardScaler()
    top_republican_trigrams['scaled_freq'] = scaler.fit_transform(top_republican_trigrams['frequency'].values.reshape(-1, 1))

    top_democrat_trigrams['scaled_freq'] = scaler.fit_transform(top_democrat_trigrams['frequency'].values.reshape(-1, 1))
    top_trigrams = pd.concat([top_democrat_trigrams.head(10),
                          top_republican_trigrams.head(10)], ignore_index=True)
    top_trigrams.sort_values(by = ['scaled_freq'], ascending = False ,inplace = True)

    fig = px.bar(top_trigrams, x='scaled_freq', y='party', 
             template='plotly_white', orientation='h',
             labels={'scaled_freq': 'Frequency of Trigram', 'trigram': 'trigram', 
                     'party': 'Party'},
             color='trigram', color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(font=dict(size=10, color='DarkSlateGray'))
    fig.update_layout(width=800, height=500)
    
    return fig.show()

def democrat_bigrams_viz(df):
    more_stopwords = ['secretary','united','states','senate','house','representative',
                   'representatives','fiscal','year','shall','adding','end','paragraph',
                   'made','available','prebody','subsection','day','date','submit','described',
                   'may','congress','following','new','enactment','code','section','assembled',
                   'b','c','amended','short','title','sec','heading', 'et', 'seq',
                    'chapter', 'effective','enacted','subchapter','entity', '42', 'usc', 'act', 'establish',
                       'categorical', 'america', '1', '2', 'seq','authorization',
                       'appropriations', 'appropriated', 'inserting','numerical',
                       'sequence','ii']
    
    democrat_words = clean_text(' '.join(df[df['party'] == 'D']['bill_text']), 
                                 more_stopwords)
    democrat_bigrams = pd.Series(nltk.ngrams(democrat_words, 2))
    top_democrat_bigrams =democrat_bigrams.value_counts().head(40)
    top_democrat_bigrams = pd.DataFrame(top_democrat_bigrams)
    top_democrat_bigrams['party'] = 'D'
    republican_words = clean_text(' '.join(df[df['party'] == 'R']['bill_text']), 
                                  more_stopwords)
    republican_bigrams = pd.Series(nltk.ngrams(republican_words, 2))
    top_republican_bigrams =republican_bigrams.value_counts().head(40)
    top_republican_bigrams = pd.DataFrame(top_republican_bigrams)
    top_republican_bigrams['party'] = 'R'
    top_democrat_bigrams.reset_index(inplace = True)
    top_republican_bigrams.reset_index(inplace = True)
    top_democrat_bigrams.rename(columns={"index": "bigram", 
                                         0: "frequency"},inplace = True)
        
    top_republican_bigrams.rename(columns={"index": "bigram", 
                                       0: "frequency"},inplace = True)
    scaler = StandardScaler()
    top_republican_bigrams['scaled_freq'] = scaler.fit_transform(top_republican_bigrams['frequency'].values.reshape(-1, 1))

    top_democrat_bigrams['scaled_freq'] = scaler.fit_transform(top_democrat_bigrams['frequency'].values.reshape(-1, 1))
    top_bigrams = pd.concat([top_democrat_bigrams.head(10), top_republican_bigrams.head(10)], 
                       ignore_index = True)
    top_bigrams.sort_values(by = ['scaled_freq'], ascending = False ,inplace = True)

    fig = px.bar(top_bigrams, x='scaled_freq', y='party', 
                 template='plotly_white', orientation='h',
                 labels={'scaled_freq': 'Frequency of Bigram', 'bigram': 'Bigram', 
                         'party': 'Party'},
                 color='bigram', color_discrete_sequence=px.colors.qualitative.Safe)
        
    fig.update_layout(font=dict(size=10, color='DarkSlateGray'))
    fig.update_layout(width=800, height=500)
    fig.show()
    
    return top_bigrams.head(2)


def partisan_viz(df):
    df['cosponsor_party'] = df['cosponsor_party'].fillna('N')
    bipart_df = pd.DataFrame(df['party'].value_counts())
    bipart_df.rename(columns = {'party': 'total_bills'}, inplace = True)
    bipart_df['bipart_bills'] = df[df['party'] != (df['cosponsor_party'])]['party'].value_counts()
    bipart_df['partisan_bills'] = df[df['party'] == (df['cosponsor_party'])]['party'].value_counts()
    bipart_df['no_cosponsor'] = df[df['cosponsor_party'] == 'N']['party'].value_counts()
    bipart_df.reset_index(inplace = True)
    bipart_df.rename(columns = {'index':'party'}, inplace = True)
    vz = bipart_df.head(2).plot(kind="bar", figsize = (5, 4), x = 'party')
    vz.set(ylabel="Number of Bills",
           xlabel="Party", title = 'Partisan vs Bipartisan Bill Breakdown')
    vz.legend(["Total Bills", "Bipartisan Bills", "Partisan Bills", "No Cosponsor"])
    
    return plt.show()

