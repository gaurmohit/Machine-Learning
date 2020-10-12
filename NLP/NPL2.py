# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:42:21 2018

@author: Raja
"""

from textblob import TextBlob
text='HMR clg is loving good'
txtbib=TextBlob(text)
senti=txtbib.sentiment.polarity
print(senti)