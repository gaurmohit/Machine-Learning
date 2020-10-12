# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:54:24 2018

@author: Raja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('C:\\Users\\Raja\\Desktop\\NLP\\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
dataset.head()
dataset.tail()

#cleaning the dataset
import re 
import nltk

#articals and terminals re called as stop words ND ARE NEED TO BE REMOVED THIS IS A STOP word
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]


for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0+i])
    review = review.lower()
    review = review.split()
    ps=PorterStemmer()
#for everry word that exists in the review then drop that word and stored in the rest of the words in the review
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)
    corpus.append(review) 
    
    