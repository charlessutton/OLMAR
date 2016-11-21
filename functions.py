# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:24:15 2016

@author: charles sutton

In this file, you will find the main filters
and basics function for time series operations
"""

import pandas as pd
import numpy as np

# Filters
# Every filter has to produce a filtered time serie of prices from a sequence of price relatives 

"""
Documentation for all the following filters : 
INPUT : 
df : a pandas.dataframe of price relatives to filter
        of shape (nb_periods , nb_shares)

        
OUTPUT :
f_df : the filtered price serie

Specific documentation added in each function
"""
    

def MA(df, window):
    """
    w : window parameter
    """
    
    p_df = to_absolute(df)
    f_ts = pd.rolling_mean(p_ts, window = window)    
    return f_ts
    
def EMA(dataset, ):    
    """pandas.ewma"""

    p_ts = to_absolute(ts)
    f_ts = pd.ewma(p_ts, )
    return f_ts
    
def ZLEMA(dataset):
    """cf filters"""
    p_ts = to_absolute(ts)
    
def KCA(dataset):
    """cf filters"""
    p_ts = to_absolute(ts)
    
# Predictive analysis
def predictions(f, dataset):
    """
    INPUT : 
    f is a filter
    dataset is a dataset of price relatives
    
    OUTPUT :
    a dataframe of the same shape than dataset, 
    With NAN values where prediction cannot be made (mostoften first days)
    And at row t+1, the prediction made in period t s.t. one can directly 
    compute metrics for performance analysis
    """

def report():
    """ see sklearn"""    
    
# Useful
def to_relative(prices):
    """
    Transfrom a price sequence to a corresponding relative sequence.
    The sequence if a pandas.series
    """
    
    

def to_absolute(price_relatives):
    """
    Transfrom a price relatives sequence to a price sequence
    The sequence if a pandas.series
    """
    return price_relatives.cumprod()
