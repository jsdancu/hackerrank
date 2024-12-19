import math
import os
import random
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import preprocessing 
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
import keras
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.models import Sequential
from keras import optimizers

def prc_CI(x, alpha):
    return np.array([np.percentile(x, 100*(alpha/2)), np.percentile(x, 100*(1-alpha/2))])

def scott_bin(x, rho, mode="nbins", alpha=0.01, EPS=1e-15):

    """ 
    Scott rule for a 2D-histogram bin widths
    
    Scott, D.W. (1992),
    Multivariate Density Estimation: Theory, Practice, and Visualization -- 2D-Gaussian case
    
    ~ N**(-1/4)

    Args:
        x     : array of 1D data (one dimension of the bivariate distribution)
        rho   : Linear correlation coefficient
        mode  : return 'width' or 'nbins'
        alpha : outlier percentile
    """
    N  = len(x)
    bw = 3.504*np.std(x)*(1 - rho**2)**(3.0/8.0)/len(x)**(1.0/4.0)

    if mode == "width":
        return bw
    else:
        return bw2bins(bw=bw, x=x, alpha=alpha)

def bw2bins(x, bw, alpha):
    """
    Convert a histogram binwidth to number of bins
    
    Args:
        x     : data array
        bw    : binwidth
        alpha : outlier percentile

    Returns:
        number of bins, if something fails return 1
    """
    if not np.isfinite(bw):
        return 1
    elif bw > 0:
        return int(np.ceil((np.percentile(x, 100*(1-alpha/2)) - np.percentile(x, 100*alpha/2)) / bw))
    else:
        return 1

def H_score(p, EPS=1E-15):
    """
    Shannon Entropy (log_e ~ nats units)

    Args:
        p : probability vector
    Returns:
        entropy
    """
    # Make sure it is normalized
    ind = (p > EPS)
    p_  = (p[ind]/np.sum(p[ind])).astype(np.float64)

    return -np.sum(p_*np.log(p_))

def I_score(C, normalized=None, EPS=1E-15):
    """
    Mutual information score (log_e ~ nats units)

    Args:
        C : (X,Y) 2D-histogram array with positive definite event counts
        normalized : return normalized version (None, 'additive', 'multiplicative')
    
    Returns:
        mutual information score
    """
    # Make sure it is positive definite
    C[C < 0] = 0

    # Joint 2D-density
    P_ij   = C / np.sum(C.flatten())

    # Marginal densities by summing over the other dimension
    P_i    = np.sum(C, axis=1); P_i /= np.sum(P_i)
    P_j    = np.sum(C, axis=0); P_j /= np.sum(P_j)
    
    # Factorized (1D) x (1D) density
    Pi_Pj  = np.outer(P_i, P_j)
    Pi_Pj /= np.sum(Pi_Pj.flatten())

    # Choose non-zero
    ind = (P_ij > EPS) & (Pi_Pj > EPS)

    # Mutual Information Definition
    I   = np.sum(P_ij[ind] * np.log(P_ij[ind] / Pi_Pj[ind]))
    I   = np.clip(I, 0.0, None)

    # Normalization
    if   normalized == None:
        return I
    elif normalized == 'additive':
        return 2*I/(H_score(P_i) + H_score(P_j))
    elif normalized == 'multiplicative':
        return I/np.sqrt(H_score(P_i) * H_score(P_j))
    else:
        raise Exception(f'I_score: Error with unknown normalization parameter "{normalized}"')

def mutual_information(x, y, weights = None, bins_x=None, bins_y=None, normalized=None,
    alpha=0.32, n_bootstrap=300,
    automethod='Scott2D', minbins=5, maxbins=100, outlier=0.01):
    """
    Mutual information entropy (non-linear measure of dependency)
    between x and y variables
    
    Args:
        x          : array of values
        y          : array of values
        weights    : weights (default None)
        bins_x     : x binning array  If None, then automatic.
        bins_y     : y binning array.
        normalized : normalize the mutual information (see I_score() function)
        n_bootstrap: number of percentile bootstrap samples
        alpha      : bootstrap confidence interval
    
    Autobinning args:    
        automethod : 'Hacine2D', 'Scott2D'
        minbins    : minimum number of bins per dimension
        outlier    : outlier protection percentile
    
    Returns:
        mutual information, confidence interval
    """

    if len(x) != len(y):
        raise Exception('mutual_information: x and y with different size.')
    
    if len(x) < 10:
        print(__name__ + f'.mutual_information: Error: len(x) < 10')
        return np.nan, np.array([np.nan, np.nan])

    x = np.asarray(x, dtype=float) # Require float for precision
    y = np.asarray(y, dtype=float)

    if weights is None:
        weights = np.ones(len(x), dtype=float)

    # Normalize to sum to one
    w = weights / np.sum(weights) 

    # For autobinning methods
    rho,_ = pearsonr(x=x,y=y)#, weights=weights)  #### not taking into account weights 

    def autobinwrap(data):
        if   automethod == 'Scott2D':
            NB = scott_bin(x=data, rho=rho, mode='nbins', alpha=outlier)
        elif automethod == 'Hacine2D':
            NB = hacine_joint_entropy_bin(x=data, rho=rho, mode='nbins', alpha=outlier)
        else:
            raise Exception(f'mutual_information: Unknown autobinning parameter <{automethod}>')

        NB = int(np.minimum(np.maximum(NB, minbins), maxbins))

        return np.linspace(np.percentile(data, outlier/2*100), np.percentile(data, 100*(1-outlier/2)), NB + 1)

    if bins_x is None:
        bins_x = autobinwrap(x)
    if bins_y is None:
        bins_y = autobinwrap(y)

    r_star = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):

        # Random values by sampling with replacement
        ind = np.random.choice(range(len(x)), size=len(x), replace=True)
        if i == 0:
            ind = np.arange(len(w))
        w_ = w[ind] / np.sum(w[ind])

        h2d = np.histogram2d(x=x[ind], y=y[ind], bins=[bins_x, bins_y])[0]#, weights=w_)[0]  #### not taking into account weights

        # Compute MI
        r_star[i] = I_score(C=h2d, normalized=normalized)
    
    # The non-bootstrapped value (original sample based)
    r    = r_star[0] 

    # Percentile bootstrap based CI
    r_CI = prc_CI(r_star, alpha)

    return r#, r_CI

def pears_matrix(df):

    """
    Pearson linear correlation matrix on dataframe df
    
    Args:
        df          : dataframe
    
    Returns:
        matrix showing the Pearson linear correlation coefficient matrix on all variables in the datframe
    """

    #size of the matrix
    m = df.shape[1]

    #declare empty matrix
    correl = np.empty((m, m), dtype=float)
    print("type(correl): ", type(correl))

    x_label = df.keys()
    y_label = df.keys()

    if evw_sign is None:
        evw_sign = np.ones(df.shape[0])

    for i, key1 in enumerate(df):
        for j, key2 in enumerate(df):
            if i > j:
                continue
            elif i == j:
                c = 1.0
            else:
                # print(f'key1: {key1}')
                # print(f'key2: {key2}')
                c = round(pearsonr(df[key1], df[key2])[0], 2)#, weights=evw_sign)[0]

            correl[i, j] = c
            correl[j, i] = c

    matrix = pd.DataFrame(correl, index=y_label, columns=x_label)
    return matrix

def mi_matrix(df, evw_sign=None):

    #size of the matrix
    m = df.shape[1]

    #declare empty matrix
    correl = np.empty((m, m), dtype=float)
    print("type(correl): ", type(correl))

    x_label = df.keys()
    y_label = df.keys()

    if evw_sign is None:
        evw_sign = np.ones(df.shape[0])

    for i, key1 in enumerate(df):
        for j, key2 in enumerate(df):
            # if i > j:
            #     continue
            # elif i == j:
            #     c = 1.0
            # else:
            #     # print(f'df[key1]: {df[key1]}')
            #     # print(f'df[key2]: {df[key2]}')
            #     c = cortools.mutual_information(df[key1], df[key2], normalized='additive')#, weights=evw_sign)

            if i > j:
                 continue
            elif i == j:
                c = 1.0
            else:
                c = mutual_information(df[key1], df[key2], normalized='additive')#, weights=evw_sign)

            correl[i, j] = c
            correl[j, i] = c

    matrix = pd.DataFrame(correl, index=y_label, columns=x_label)
    return matrix

def preprocessing_df(df):

	scaler = preprocessing.StandardScaler().fit(df)
	df_scaled = scaler.transform(df)

	return df_scaled

def multilinear_regression(X_train, y_train):

    #### Creating a regression model 
    reg = LinearRegression() 
    
    #### Fitting the regression model
    reg.fit(X_train, y_train)  

    X_train2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X_train2)
    print(est.fit().summary())

    return reg

def rsquared_score(y_true, y_pred):

    sum_squares_residuals = sum(np.square(y_true - y_pred))
    sum_squares = sum(np.square(y_true - statistics.mean(y_true)))
    r2score = 1.0 - sum_squares_residuals / sum_squares

    return r2score

def dNN_regression(X_train, y_train):

    model_dNN = Sequential()

    inputFeatures = keras.layers.Input(shape=(5,))
    print(inputFeatures)
    dnn = keras.layers.Dense(4, activation='relu')(inputFeatures)
    print(dnn)
    # dnn = keras.layers.Dropout(0.1)(dnn)
    # print(dnn)
    dnn = keras.layers.Dense(4, activation='relu')(dnn)
    # print(dnn)
    # dnn = keras.layers.Dropout(0.1)(dnn)
    # print(dnn)

    predictFraction = keras.layers.Dense(1)(dnn)
    print(predictFraction)

    model_dNN = keras.models.Model(inputs=[inputFeatures],outputs=[predictFraction])

    initial_learning_rate = 0.1 
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                        decay_steps=10, decay_rate=0.95, staircase=True)

    model_dNN.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss='mean_absolute_error')

    history = model_dNN.fit(X_train, y_train, epochs=1000, verbose=0, validation_split = 0.2)

    return model_dNN, history


if __name__ == '__main__':

    ###### Read in data sets from the Excel spreadsheet ###########
    data_daily = pd.read_excel(open('case.xlsx', 'rb'), sheet_name='data_daily')
    data_monthly = pd.read_excel(open('case.xlsx', 'rb'), sheet_name='data_monthly')
    data_intraday = pd.read_excel(open('case.xlsx', 'rb'), sheet_name='data_intraday')

    # print(data_daily)
    # print(data_monthly)
    # print(data_intraday)

    ########## Part 1: visualise data ##############

    ######### Plot first data set (data_daily) ############
    fig, axes = plt.subplots(figsize=(10, 6))
    plt.plot(data_daily['Dates'], data_daily['Soybean'], label='Soybean')
    plt.plot(data_daily['Dates'], data_daily['Corn'], label='Corn')
    plt.plot(data_daily['Dates'], data_daily['S&P500'], label='S&P500')
    plt.xticks(rotation=45)
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.title("Equity prices vs time")
    axes.legend(loc="right", bbox_to_anchor=(0., 0.3, 1., 0.7))
    plt.savefig("data_daily1.pdf")
    plt.clf()

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.plot(data_daily['Dates'], data_daily['CrudeOil'], label='CrudeOil')
    plt.plot(data_daily['Dates'], data_daily['DXY'], label='DXY')
    plt.xticks(rotation=45)
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.title("Equity prices vs time")
    axes.legend(loc="right", bbox_to_anchor=(0., 0.3, 1., 0.7))
    plt.savefig("data_daily2.pdf")
    plt.clf()

    ########## Plot second data set (data_monthly) ################
    fig, axes = plt.subplots(figsize=(10, 6))
    plt.plot(data_monthly['Dates'], data_monthly['stock_to_use'])
    plt.xticks(rotation=45)
    plt.xlabel("Dates")
    plt.ylabel("Stock to use")
    plt.title("Stock to use vs time")
    plt.savefig("data_monthly.pdf")
    plt.clf()


    ########## Part 2: compare time series #############

    ##### Drop day from the dates in data_monthly ######
    data_monthly1 = data_monthly.copy()
    data_monthly1['Dates'] = data_monthly['Dates'].dt.to_period('M')

    ###### Calculating average price per month for soybean and S&P500 #######
    df_soybean_monthly_avg = data_daily.groupby(pd.PeriodIndex(data_daily['Dates'], freq="M"))['Soybean'].mean().reset_index()
    df_sp500_monthly_avg = data_daily.groupby(pd.PeriodIndex(data_daily['Dates'], freq="M"))['S&P500'].mean().reset_index()

    ###### Making a new dataframe where I add the monthly values for soybean, S&P500 and stock to use #####
    df_monthly_avg = df_soybean_monthly_avg.merge(df_sp500_monthly_avg, on='Dates', how='right')
    df_monthly_avg = df_monthly_avg.merge(data_monthly1, on='Dates', how='right')

    #print(df_monthly_avg)

    ##### Plotting in 2D scatter plots the various equities against each other in pair ####
    fig, axes = plt.subplots(figsize=(10, 6))
    plt.scatter(df_monthly_avg['Soybean'], df_monthly_avg['stock_to_use'])
    plt.xlabel("Soybean (monthly average)")
    plt.ylabel("Stock to use")
    plt.title("Soybean prices vs stock to use")
    plt.savefig("soybean_stock_to_use.pdf")
    plt.clf()

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.scatter(df_monthly_avg['S&P500'], df_monthly_avg['stock_to_use'])
    plt.xlabel("S&P500 (monthly average)")
    plt.ylabel("Stock to use")
    plt.title("S&P500 prices vs stock to use")
    plt.savefig("sp500_stock_to_use.pdf")
    plt.clf()

    # fig, axes = plt.subplots(figsize=(10, 6))
    # plt.scatter(data_daily['Soybean'], data_daily['S&P500'])
    # plt.xlabel("Soybean")
    # plt.ylabel("S&P500")
    # plt.title("Soybean prices vs S&P500")
    # plt.savefig("soybean_sp500.pdf")
    # plt.clf()

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.scatter(df_monthly_avg['Soybean'], df_monthly_avg['S&P500'])
    plt.xlabel("Soybean (monthly average)")
    plt.ylabel("S&P500 (monthly average)")
    plt.title("Soybean prices vs S&P500 (monthly average)")
    #axes.legend(pearsonr(df_monthly_avg['Soybean'], df_monthly_avg['S&P500'])[0])
    plt.savefig("soybean_sp500.pdf")
    plt.clf()

    ########## Part 3: multilinear regression #############

    #### Soybean is what we want to predict, i.e. y-value ####
    #### Use other available variables to do multilinear regression ####

    ### Add Crude oil, corn, DXY to df_monthly_avg ###
    df_corn_monthly_avg = data_daily.groupby(pd.PeriodIndex(data_daily['Dates'], freq="M"))['Corn'].mean().reset_index()
    df_crudeoil_monthly_avg = data_daily.groupby(pd.PeriodIndex(data_daily['Dates'], freq="M"))['CrudeOil'].mean().reset_index()
    df_dxy_monthly_avg = data_daily.groupby(pd.PeriodIndex(data_daily['Dates'], freq="M"))['DXY'].mean().reset_index()

    df_monthly_avg = df_monthly_avg.merge(df_corn_monthly_avg, on='Dates', how='right')
    df_monthly_avg = df_monthly_avg.merge(df_crudeoil_monthly_avg, on='Dates', how='right')
    df_monthly_avg = df_monthly_avg.merge(df_dxy_monthly_avg, on='Dates', how='right')

    ##### Remove outlier from stock to use ######
    df_monthly_avg = df_monthly_avg.loc[df_monthly_avg['stock_to_use']<0.2]

    ##### Evaluate pearson correlation coefficient between variables ####
    df_monthly_avg = df_monthly_avg.dropna()
    pears1 = round(pearsonr(df_monthly_avg['Soybean'], df_monthly_avg['stock_to_use'])[0], 2)
    print('pearson corr. coeff. soybean vs stock to use: ', pears1)
    pears2 = round(pearsonr(df_monthly_avg['S&P500'], df_monthly_avg['stock_to_use'])[0], 2)
    print('pearson corr. coeff. s&p500 vs stock to use: ', pears2)
    pears3 = round(pearsonr(df_monthly_avg['Soybean'], df_monthly_avg['S&P500'])[0], 2)
    print('pearson corr. coeff. soybean vs s&p500: ', pears3)

    #### Calculating all Pearson linear coefficients for all variable combinations, saving them into a matrix and plot it #####
    pears_corr_matrix = df_monthly_avg[['Soybean', 'S&P500', 'stock_to_use', 'Corn', 'CrudeOil', 'DXY']].corr(method='pearson')
    #pears_corr_matrix = pears_matrix(df_monthly_avg[['Soybean', 'S&P500', 'stock_to_use', 'Corn', 'CrudeOil', 'DXY']])

    fig, axes = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.15)
    plt.title("Pearson correlation matrix (monthly average)")
    g = sns.heatmap(pears_corr_matrix, annot=True, cmap=plt.cm.Reds, fmt=".2f")
    plt.savefig("pears_corr_all_vars.pdf")
    plt.clf()

    #### Calculating all Spearman's rank correlation coefficients for all variable combinations, saving them into a matrix and plot it #####
    spearman_corr_matrix = df_monthly_avg[['Soybean', 'S&P500', 'stock_to_use', 'Corn', 'CrudeOil', 'DXY']].corr(method='spearman')

    fig, axes = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.15)
    plt.title("Spearman's rank correlation matrix (monthly average)")
    g = sns.heatmap(spearman_corr_matrix, annot=True, cmap=plt.cm.Reds, fmt=".2f")
    plt.savefig("spearman_corr_all_vars.pdf")
    plt.clf()

    #### Calculating all Pearson linear coefficients for all variable combinations, saving them into a matrix and plot it #####
    mi_corr_matrix = mi_matrix(df_monthly_avg[['Soybean', 'S&P500', 'stock_to_use', 'Corn', 'CrudeOil', 'DXY']])

    fig, axes = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.15)
    plt.title("Mutual information correlation matrix (monthly average)")
    g = sns.heatmap(mi_corr_matrix, annot=True, cmap=plt.cm.Reds, fmt=".2f")
    plt.savefig("mi_corr_all_vars.pdf")
    plt.clf()

    ##### Create 2D correlation scatter plots for all variables ####
    g = sns.pairplot(df_monthly_avg)
    g.savefig('correlation_plots_all_vars.pdf')

    #### Saving Soybean into a separate dataframe as it's the y-value and dropping that column from df_monthly_avg ####
    y = df_monthly_avg['Soybean']
    df_monthly_avg = df_monthly_avg.drop('Soybean', axis=1)

    #### Only using the equities for the regression task, not the dates ####
    relevant_vars = ['S&P500', 'stock_to_use', 'Corn', 'CrudeOil', 'DXY']
    df_monthly_avg1 = df_monthly_avg[relevant_vars].copy()

    #### Preprocessing dataframe to be on sclae of [0, 1] to make linear regression more robust ####
    df_monthly_avg1 = preprocessing_df(df_monthly_avg1)

    #### Creating train and test sets ###
    X_train, X_test, y_train, y_test = train_test_split(df_monthly_avg1, y, test_size=0.2) 

    predictions = {}

    #### Defining the multilinear model ####
    linear_model = multilinear_regression(X_train, y_train)

    #### Making predictions 
    predictions['multilinear'] = linear_model.predict(X_test) 

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, predictions['multilinear'])
    plt.title('Multilinear model truth values vs predicted')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.savefig('linear_model_true_pred.pdf')

    #### Model evaluation ####
    print('multilinear mean_squared_error: ', round(mean_squared_error(y_test, predictions['multilinear']), 2)) 
    print('multilinear mean_absolute_error: ', round(mean_absolute_error(y_test, predictions['multilinear']), 2))

    #### Model coefficients & intercept ###
    print('multilinear model coeff.: ', linear_model.coef_) 
    print('multilinear model intercept: ', round(linear_model.intercept_, 2))

    #### R-squared score #####
    r2score = rsquared_score(y_test.to_numpy(), predictions['multilinear'].flatten())
    print('multilinear R-squared score: ', round(r2score, 2))

    ########## Part 4: small dense NN #############
    
    ##### Defining the small dense neural network ####
    dnn_model, dnn_model_history = dNN_regression(X_train, y_train)

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.plot(dnn_model_history.history['loss'], label='train loss')
    plt.plot(dnn_model_history.history['val_loss'], label='test loss')
    plt.title('Dense NN model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    axes.legend(loc='right')
    plt.savefig('nn_model_loss.pdf')

    #### Making predictions #########
    predictions['dnn'] = dnn_model.predict(X_test)

    #### Model evaluation #####
    print('dense NN mean_squared_error: ', round(mean_squared_error(y_test, predictions['dnn']), 2)) 
    print('dense NN mean_absolute_error (also test loss): ', round(mean_absolute_error(y_test, predictions['dnn']), 2))

    #### R-squared score #####
    # r2score = keras.metrics.R2Score().update_state(np.array(y_test, dtype=np.float32), np.array(predictions['dnn'], dtype=np.float32))
    # r2score_result = r2score.result()
    r2score = rsquared_score(y_test.to_numpy(), predictions['dnn'].flatten())
    print('dense NN R-squared score: ', round(r2score, 2))

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, predictions['dnn'])
    plt.title('Dense NN model truth values vs predicted')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.savefig('nn_model_true_pred.pdf')

    
    ########## Part 7: intraday data set #############

    ######### Plot first data set (data_intraday) ############
    fig, ax1 = plt.subplots(figsize=(10, 9))
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlabel("Intraday index")
    ax1.set_ylabel("Prices")
    plt.title("Equity prices vs time")
    ax1.set_ylim(41, 47)
    p1 = ax1.plot(data_intraday['index_brent'], data_intraday['brent'], label='brent', color='blue')
    ax2 = ax1.twinx()
    ax2.set_ylim(3000, 3500)
    p2 = ax2.plot(data_intraday['index_sp500'], data_intraday['sp500'], label='S&P500', color='orange')

    ax1.legend(handles=p1+p2, loc="lower right")
    plt.savefig("data_intraday.pdf")
    plt.clf()

    ##### Creating a new dataframe where the brent and S&P500 are matched by intraday index value ####
    data_intraday1 = data_intraday[['index_brent', 'brent']].copy().rename(columns={'index_brent':'index'})
    data_intraday2 = data_intraday[['index_sp500', 'sp500']].copy().rename(columns={'index_sp500':'index'})

    data_intraday1 = data_intraday1.merge(data_intraday2, on='index', how='right')
    data_intraday1 = data_intraday1.dropna()
    #print(data_intraday1)

    ######### Plot intraday data set matched values only ############
    fig, ax1 = plt.subplots(figsize=(10, 9))
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlabel("Intraday index")
    ax1.set_ylabel("Prices")
    plt.title("Equity prices vs time")
    ax1.set_ylim(41, 47)
    p1 = ax1.plot(data_intraday1['index'], data_intraday1['brent'], label='brent', color='blue')
    ax2 = ax1.twinx()
    ax2.set_ylim(3000, 3500)
    p2 = ax2.plot(data_intraday1['index'], data_intraday1['sp500'], label='S&P500', color='orange')

    ax1.legend(handles=p1+p2, loc="lower right")
    plt.savefig("data_intraday1.pdf")
    plt.clf()

    ##### Plotting in 2D scatter plot brent vs S&P500 ####
    fig, axes = plt.subplots(figsize=(10, 6))
    plt.scatter(data_intraday1['sp500'], data_intraday1['brent'])
    plt.xlabel("S&P500 price")
    plt.ylabel("Brent price")
    plt.title("Brent price vs S&P500 price")
    plt.savefig("brent_sp500.pdf")
    plt.clf()

    ###### Calculating the Pearson linear coeff. between brent and S&P500 ####
    pears_intraday = round(pearsonr(data_intraday1['brent'], data_intraday1['sp500'])[0], 2)
    print('pearson corr. coeff. brent vs S&P500: ', pears_intraday)

    ##### Calculating the mutual information coefficient between brent and S&P500 ####
    mi_intraday = round(mutual_information(data_intraday1['brent'], data_intraday1['sp500'], normalized='additive'), 2)
    print('mutual information corr. coeff. brent vs S&P500: ', mi_intraday)