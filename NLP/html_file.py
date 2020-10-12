# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:31:07 2018

@author: Raja
"""

from bs4 import BeautifulSoup
html_file=open('C:\\Users\\Raja\\Desktop\\NLP\\ab.html','r')
page=html_file.read()
soup=BeautifulSoup(page,'html.parser')

reviews=soup.find_all('p')
for p in reviews:
    print(p.get_text())
    
    
from textblob import TextBlob
positive, negative=0,0
for p in reviews:
    text=p.get_text()
    sentiment=TextBlob(text).sentiment.polarity
    if(sentiment>=0):
        positive+=1
    else:
        negative+=1
        
print("positive review :", positive)
print("negative review :", negative)
