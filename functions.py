# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:24:15 2016

@author: charles sutton

In this file, you will find the main filters
and basics function for time series operations
"""

# Filters
# Every filter has to produce a filtered time serie of prices from a sequence of price relatives 
"""
Documentation for all the following filters : 
INPUT : 
dataset : a pandas.dataframe of price relatives to filter
        of shape (nb_periods , nb_shares)
params : a dict of parameters
        
OUTPUT :
f_dataset : the filtered price serie dataset
The resulting dataframe has the following properties : 

    - same shape than the original dataset, 
    - NAN values where prediction cannot be made (mostoften first days)
    - And at row t, the prediction made in period t 
    - Note that you have to adjust the dataframe to perform predictive analysis
      otherwise you are measuring with knowledge of future information.


Specific documentation added in each function
""" 
def MA(dataset, params):
    """
    Moving average
    params should at least contain 
    window : window parameter
    """
    assert "window" in params, "you should add the window parameter"
    p_dataset = to_absolute(dataset)
    f_dataset = p_dataset.rolling(window = params["window"]).mean()    
    return f_dataset
    
def EMA(dataset, params):    
    """
    Exponential moving average
    params should at least contain 
    com : is the center of mass parameter
    """
    assert "com" in params, "you should add the com (center of mass) parameter"
    p_dataset = to_absolute(dataset)
    f_dataset = p_dataset.ewm(com=params["com"]).mean()
    return f_dataset
    
def ZLEMA(dataset, params):
    """cf filters"""
#    p_dataset = to_absolute(dataset)
    
def KCA(dataset):
    """cf filters"""
#    p_dataset = to_absolute(dataset)
    
# Predictive analysis
    
def adjust_data(dataset, prediction ,horizon = 1):
    """
    Aims to adjust the prediction and the real price relative for the measure of performance
    you can adjust the horizon.
    """

    assert dataset.shape == prediction.shape
        
    adjusted_prediction = prediction[:-horizon].dropna(axis=0, how='all', inplace=False)
    starting_idx = adjusted_prediction.index[0]    
    adjusted_dataset = dataset[starting_idx+horizon:]

    assert adjusted_dataset.shape == adjusted_prediction.shape

    return adjusted_dataset, adjusted_prediction
    
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd 

def regression_report(adjusted_dataset, adjusted_prediction, output="all"):
    """ 
    Build a regression task report for the adjusted datasets
    report includes 
    MAE : mean average error
    R2 : r2-score  
    DPA : direction prediction accuracy error
    
    if output = "all" then it outputs the report stock per stock and the average
    if output = "average" the it ouputs the average only
    """

    df = pd.DataFrame()

    df["MAE"] = np.insert(mean_absolute_error(adjusted_dataset, adjusted_prediction,multioutput = "raw_values"), 0 , mean_absolute_error(adjusted_dataset, adjusted_prediction))
    df["DPA"] = direction_prediction_accuracy(adjusted_dataset,adjusted_prediction)
    df["R2"] = np.insert(r2_score(adjusted_dataset, adjusted_prediction,multioutput = "raw_values"), 0 , r2_score(adjusted_dataset, adjusted_prediction))

    # setting stock names as index
    df.index = adjusted_dataset.columns.insert(0, u'uniform_average')

    if output == "all" :
        return df
    elif output == "average" :
        return df[0]

def direction_prediction_accuracy(adjusted_dataset, adjusted_prediction):
    """
    compute direction prediction accuracy
    """
    multi = (np.asanyarray(adjusted_dataset)-1)*(np.asanyarray(adjusted_prediction)-1)
    direction_success = np.zeros(multi.shape)
    direction_success[multi>=0] = 1.0 #positive element are denoting same direction prediction !
    raw_values = np.mean(direction_success, axis = 0)
    uniform_average = np.mean(raw_values)
    return np.insert(raw_values,0,uniform_average)

# Useful

def to_absolute(price_relatives):
    """
    Transfrom a price relatives dataframe to a price sequence dataframe
    The sequence if a pandas.series
    """
    return price_relatives.cumprod(axis=0)
