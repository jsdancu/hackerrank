import math
import os
import random
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy import stats
import argparse

def carino(df, period='', factor='COUNTRY'):

    #### Pivot dataframe by the Factor Selection: COUNTRY, SECTOR or TICKER
    if factor in ['COUNTRY', 'SECTOR', 'TICKER']:
        df = df.pivot(index=factor, columns='REF_DATE', values='TOTAL')
    else:
        print('Factor Selection not defined')
        return []

    #### Handling NaN values in dataframe: replacing them with 0 ######
    df = df.fillna(0)

    #### Calculate single period returns r_i ####
    single_period_return = df.sum(axis=0)

    #### Calculate multi-period return R ####
    R = np.prod(1.0 + single_period_return) - 1.0
    print(f"Total multi period return R: {R*100:.2f}%")
    
    #### Calculate multi-period adjustment P ####
    P = math.log(1.0 + R)/R
    print(f"P: {P}")

    #### Calculate single-period adjustment A_i ####
    A = np.log(1.0 + single_period_return)/single_period_return
    print(f"A: {A}")

    #### Calculate the Carino smoothed dataframe ####
    df = A/P * df

    #### Return dataframe for defined period #####
    if period in ['Q', 'Y']:
        df_period = df.T.groupby(pd.PeriodIndex(df.columns, freq=period)).sum().T
    elif period == 'I':
        df_period = df.sum(axis=1)
    else:
        print('Period not defined')
        return df, []

    return df, df_period


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', dest='data_file', action='store',default="CodingTestData.xlsx")
    parser.add_argument('--period', dest='period', action='store', default="Q") # can be Q=quaterly, Y=yearly or I=inception to date
    parser.add_argument('--factor', dest='factor', action='store', default="COUNTRY") #can be SECTOR or TICKER
    
    args = parser.parse_args()

    ###### Read in data set from the Excel spreadsheet ###########
    try:
        f = open(args.data_file, 'rb')
        data = pd.read_excel(f, sheet_name='Sheet1')
    except:
        print('Error opening file/loading data')

    df, df_period = carino(data, args.period, args.factor)
    print(df_period)

    #### Save to Excel file ####
    df_period.to_excel("output.xlsx") 