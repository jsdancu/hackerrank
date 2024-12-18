import math
import os
import random
import re
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def Pearson_corr_coeff(X, y):

    n = len(X)
    sum_X = sum(X)
    sum_y = sum(y)
    
    # mean of x and y vector
    m_X = sum_X/n
    m_y = sum_y/n

    cov_Xy = 0
    std_X = 0
    std_y = 0

    for i in range(n):
        cov_Xy += (X[i]-m_X)*(y[i]-m_y)
        std_X += (X[i]-m_X)**2
        std_y += (y[i]-m_y)**2

    std_X = math.sqrt(std_X)
    std_y = math.sqrt(std_y)

    rho = cov_Xy/(std_X*std_y)

    return rho

def LinearRegression(X, y):
    
    # number of observations/points
    n = len(X)
    sum_X = sum(X)
    sum_y = sum(y)
    sum_Xy = 0
    sum_XX = 0
    
    for i in range(n):
        sum_Xy += X[i]*y[i]
        sum_XX += X[i]**2
    
    # mean of x and y vector
    m_X = sum_X/n
    m_y = sum_y/n
    
    # calculating cross-deviation and deviation about x
    SS_xy = sum_Xy - n*m_y*m_X
    SS_xx = sum_XX - n*m_X*m_X
    
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_X

    return (b_0, b_1)

if __name__ == '__main__':
    # grades = pd.DataFrame({'physics': [15, 12,8, 8, 7, 7, 7, 6, 5, 3], 'history': [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]})
    # print(grades)

    # plt.plot(grades['physics'], grades['history'], '.')
    # plt.xlabel("Physics")
    # plt.ylabel("History")
    # plt.title("Physics vs History grades")
    # #plt.show()
    # plt.savefig("physics_history.pdf")

    # reg = LinearRegression().fit(grades['physics'].values.reshape(-1, 1), grades['history'].values.reshape(-1, 1))

    # physics = [[10]]
    # history = reg.predict(physics)
    # print(round(history[0][0], 1))

    physics = [15, 12,8, 8, 7, 7, 7, 6, 5, 3]
    history = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]
    
    phys = int(input())
    
    reg = LinearRegression(physics, history)

    print(f'{reg[1]:.3f}')
    
    hist_pred = reg[0] + reg[1]*phys
    print(round(hist_pred, 1))

    rho = Pearson_corr_coeff(physics, history)
    print(f'{rho:.3f}')

