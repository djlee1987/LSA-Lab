
# coding: utf-8

# In[1]:

"""
Created on Thu Feb 02 20:01:17 2017
Python 3.5.2 |Anaconda 4.2.0 (64-bit)

@author: djlee1987 Daniel Lee
UIS - Data Science Essentials
LSA Lab
"""


# In[2]:

### import all necessary modules
import pandas as pd
#from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups


# In[3]:

categories = ['sci.space']
dataset = fetch_20newsgroups(subset='all',shuffle=True, random_state=42, categories=categories)
corpus = dataset.data


# In[4]:

stopset = set(stopwords.words('english'))
stopset.update(['lt','p','/p','br','amp','quot','field','font','span','0px','rgb','style','51', 
                'spacing','text','helvetica', 'family', 'arial', 'indent', 'letter'
                'line','none','sans','serif','transform','variant','strong', 'video', 'title'
                'white','word','letter', 'roman','0pt','16','color','12','14','21', 'neue', 'apple', 'class',
                'edu', 'com', 'gov', 'net', '00', '000', '0000', '___', '__', '00000', '000000', '000021',
                'nntp', '00041032', '000062david42', '000050', '00041555', '0004244402', 'mcimail', '00043819',
                'prb', '0004246', '0004422', '00044513', '00044939','access', 'digex', 'host', 'would', 'writes',
                'posting', 'dseg'])


# In[5]:

vectorizer = TfidfVectorizer(stop_words=stopset,
                                use_idf=True, ngram_range = (1, 3))
X = vectorizer.fit_transform(corpus)


# In[6]:

#decompose into X=UST^T
lsa = TruncatedSVD(n_components = 25, n_iter = 100)
lsa.fit(X)


# In[7]:

terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_):
    termsInComp = zip (terms, comp)
    sortedTerms = sorted(termsInComp, key = lambda x: x[1], reverse = True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print(" ")


# ### issues / need to dive-deep/research possible solutions
# <li> Need to balance # of concepts to ensure sufficient differentiation between concepts without over dilution of insights per concept
# <li> Number starting '000' presents an issue integrating into stopwords.  startwith() -like method available for use in stopset?
# 
