# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:13:44 2018

@author: Raja
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm

digits=load_digits()
plt.gray()
plt.matshow(digits.images[20])  # in o/p it always shows last digits here shows 0 if there will be 16 then o/p is 6
plt.show()
print(digits.images[20])

clf=svm.SVC()

#train the model
clf.fit(digits.data[:-1],digits.target[:-1])

#test the model
prediction = clf.predict(digits.data[20:21])

print('predicted',prediction)