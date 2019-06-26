
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.compose import TransformedTargetRegressor

def aggregate_last_n_hours(df_hour,instant,n_hours):
    agg = 0
    for i in range(1,n_hours+1):
        if (instant-i <=0) or not(instant - i in df_hour.index):
            agg =+ df_hour["cnt"].mean()
        else:
            #The cnt of the instant IS EXCLUDED (.loc[instant-1-i]). That is important to avoid data leakeage
            current_index = instant-1
            agg =+df_hour["cnt"].loc[ current_index - i ]
    return agg/n_hours

def one_hot_encode(df,categorical2encode):
    for cat_var in categorical2encode:
        df = pd.concat([df,pd.get_dummies(df[cat_var],prefix=cat_var)], axis=1)
        df.drop(columns=cat_var,inplace=True)
    print("Number of model features after one-hot encoding: {}".format(len(df.columns)-1))
    return df

def split_train_test(df,fraction_train=0.8):
    train_set = df.loc[ : int(fraction_train*len(df)) -1]
    test_set =  df.loc[ int(fraction_train*len(df)) : ]
    y_train = train_set.pop("cnt")
    X_train = train_set
    y_test = test_set.pop("cnt")
    X_test = test_set
    return X_train, y_train, X_test, y_test

def compare_predictions_vs_true(y_train_pred,y_train, y_test_pred,y_test,saveplot=""):  
    plt.figure(figsize=(8,5))
    label_train = r'Train predictions: $R^2$=%.2f, $MAE$=%.2f, $MAD$=%.2f' % ( r2_score(y_train.values, y_train_pred), 
            median_absolute_error(y_train.values, y_train_pred), pd.Series(y_train_pred).mad() )
    ax1= sns.scatterplot(x=y_train.values,y=y_train_pred,label=label_train,size=0.2,color="r",alpha=0.5)
    label_test = r'Test predictions: $R^2$=%.2f, $MAE$=%.2f, $MAD$=%.2f' % ( r2_score(y_test.values, y_test_pred),
            median_absolute_error(y_test.values, y_test_pred), pd.Series(y_test.mad()) )
    sns.scatterplot(x=y_test.values,y=y_test_pred,size=0.4,label=label_test,color="b",alpha=0.4,ax=ax1)
    ax1.plot([0, 1000], [0, 1000], '--k')
    ax1.set( ylabel='Predicted target', xlabel='True Target')
    handles, labels = ax1.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if i in [0,5]]
    ax1.legend(*zip(*unique), ncol=1, loc="upper left", frameon=True)
    plt.tight_layout()
    if saveplot !="":
        plt.savefig("./report/{}.png".format(saveplot),format='png')
    return

def plot_residuals(y_true,y_pred, saveplot=""):
    plt.figure(figsize=(9,5.5))
    for i in y_pred.keys():
        label = "{}: ".format(i)+r'$R^2$=%.2f, $MAE$=%.2f' % ( r2_score(y_true.values, y_pred[i]), 
                                                              median_absolute_error(y_true.values, y_pred[i]))
        residuals = y_true.values - y_pred[i]
        sns.scatterplot(x=y_true,y=residuals,label=label,size=0.2,alpha=0.3)
    ax1=plt.gca()
    ax1.plot([1000, 0], [0,0], '--k')
    ax1.set( ylabel='Residuals (= True - Predicted)', xlabel='cnt', ylim=(-850,650),xlim=(-10,950))
    handles, labels = ax1.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if i%5==0]
    ax1.legend(*zip(*unique), ncol=1, loc="lower left", frameon=True,framealpha=1.0)
    ax1.grid(axis="y",linestyle='--', linewidth=1.0)
    plt.tight_layout()
    if saveplot !="":
        plt.savefig("./report/{}.png".format(saveplot),format='png')
    return residuals

