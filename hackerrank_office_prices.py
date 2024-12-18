import math
import os
import random
import re
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':

    f, n = map(int, input().rstrip().split())
    
    X = []
    y = []

    for i in range(n):
        xy = list(map(float, input().rstrip().split()))
        X.append(xy[:-1])
        y.append(xy[-1])
    
    X = np.array(X)
    y = np.array(y)
    # print(x)
    # print(y)
    
    poly = PolynomialFeatures(degree=3)

    Xpoly = poly.fit_transform(X)

    reg = LinearRegression().fit(Xpoly, y)
        
    t = int(input())
    Xtest = []
    for i in range(t):
        Xtest.append(list(map(float, input().rstrip().split())))
    
    Xtest = np.array(Xtest)
    
    Xtestpoly=poly.transform(Xtest)
    
    predicted = reg.predict(Xtestpoly)
    
    for p in predicted:
        print(f'{p:.2f}')