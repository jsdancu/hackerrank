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

def calc_cash(m, buy, buy_price, sell, sell_price):

    cash = m - buy*buy_price + sell*sell_price

    return cash

def printTransactions(m, k, stocks):

    t = np.array(range(5))

    reg1 = LinearRegression().fit(t.reshape(-1, 1), stocks['iStreet']['prices'].reshape(-1, 1))
    reg2 = LinearRegression().fit(t.reshape(-1, 1), stocks['HR']['prices'].reshape(-1, 1))

    y1 = reg1.predict(t.reshape(-1, 1))
    y2 = reg2.predict(t.reshape(-1, 1))

    '''
    plt.scatter(t, stocks['iStreet']['prices'], label = "iStreet")
    plt.scatter(t, stocks['HR']['prices'], label = "HR")
    plt.plot(t, y1)
    plt.plot(t, y2)
    plt.xlabel("days")
    plt.ylabel("prices")
    plt.title("Prices vs time")
    plt.legend()
    #plt.show()
    plt.savefig("stock_predictions.pdf")

    print(reg1.coef_[0][0], reg1.intercept_)
    print(reg2.coef_[0][0], reg2.intercept_)
    '''

    if reg1.coef_[0][0] > reg2.coef_[0][0]:
        buy = m // stocks['iStreet']['prices'][-1]
        sell = stocks['HR']['owned']
        print(buy+sell)
        print(f'iStreet BUY {buy}')
        print(f'HR SELL {sell}')

        stocks['HR']['owned'] = 0
        stocks['iStreet']['owned'] += buy

        return calc_cash(m, buy, stocks['iStreet']['prices'][-1], sell, stocks['HR']['prices'][-1])

    else:
        buy = m // stocks['HR']['prices'][-1]
        sell = stocks['iStreet']['owned']
        print(buy+sell)
        print(f'iStreet SELL {sell}')
        print(f'HR BUY {buy}')

        stocks['HR']['owned'] += buy
        stocks['iStreet']['owned'] = 0

        return calc_cash(m, buy, stocks['HR']['prices'][-1], sell, stocks['iStreet']['prices'][-1]) 



def printTransactions_dummy(m, k, stocks):

    t = np.array(range(3))

    for i in range(3):

        reg1 = LinearRegression().fit(t.reshape(-1, 1), stocks['iStreet']['prices'][i:i+3].reshape(-1, 1))
        reg2 = LinearRegression().fit(t.reshape(-1, 1), stocks['HR']['prices'][i:i+3].reshape(-1, 1))

        y1 = reg1.predict(t.reshape(-1, 1))
        y2 = reg2.predict(t.reshape(-1, 1))

        if reg1.coef_[0][0] > reg2.coef_[0][0]:
            buy = m // stocks['iStreet']['prices'][i+2]
            sell = stocks['HR']['owned']
            print(buy+sell)
            print(f'iStreet BUY {buy}')
            print(f'HR SELL {sell}')

            stocks['HR']['owned'] = 0
            stocks['iStreet']['owned'] += buy

            m = calc_cash(m, buy, stocks['iStreet']['prices'][i+2], sell, stocks['HR']['prices'][i+2])
            print(m)

        else:
            buy = m // stocks['HR']['prices'][i+2]
            sell = stocks['iStreet']['owned']
            print(buy+sell)
            print(f'iStreet SELL {sell}')
            print(f'HR BUY {buy}')

            stocks['HR']['owned'] += buy
            stocks['iStreet']['owned'] = 0

            m = calc_cash(m, buy, stocks['HR']['prices'][i+2], sell, stocks['iStreet']['prices'][i+2]) 
            print(m)

    return m + stocks['iStreet']['prices'][-1] * stocks['iStreet']['owned'] + stocks['HR']['prices'][-1] * stocks['HR']['owned'] 
          

if __name__ == '__main__':
    
    m, k, d = map(int, input().rstrip().split()) #m=available money; k=no. of different stocks available; d=number of remaining trading days
    stocks = {}
    for i in range(k):
        line = input().rstrip().split() #name, owned, prices
        name = line[0]
        owned = int(line[1])
        #prices = [float(x) for x in line[2:]]
        prices = list(map(float, line[2:]))

        stocks[name] = {}
        stocks[name]['owned'] = owned
        stocks[name]['prices'] = np.array(prices)

    # print(name)
    # print(owned)
    # print(prices)

    m = printTransactions(m, k, stocks)
    print(m)

    '''
    stocks = {}
    with open(os.path.join("bigger_data_set.txt")) as f:
        for l in f:
            line = l.rstrip().split()
            stocks[line[0]] = list(map(float, line[1:]))           
    
    #print(list(stocks.keys())[0])

    t = np.array(range(len(stocks[list(stocks.keys())[0]])))

    # plt.scatter(t, stocks['iStreet']['prices'], label = "iStreet")
    # plt.scatter(t, stocks['HR']['prices'], label = "HR")
    for stock in stocks.keys():
        plt.plot(t, stocks[stock], label = stock)
    plt.xlabel("days")
    plt.ylabel("prices")
    plt.title("Prices vs time")
    plt.legend()
    #plt.show()
    plt.savefig("stock_predictions_big.pdf")
    '''
    


