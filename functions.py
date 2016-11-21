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
dataset : a pandas.dataframe of price relatives to filter
        of shape (nb_periods , nb_shares)
params : a dict of parameters
        
OUTPUT :
f_dataset : the filtered price serie dataset
The resulting dataframe is such that 

Specific documentation added in each function
"""
    
def MA(dataset, params):
    """
    Moving average
    params should at least contain 
    w : window parameter
    """
    p_dataset = to_absolute(dataset)
    f_dataset = pd.rolling_mean(p_dataset, window = params["window"])    
    return f_dataset
    
def EMA(dataset, params):    
    """
    Exponential moving average
    params should at least contain 
    com : is the center of mass parameter
    """
    p_dataset = to_absolute(dataset)
    f_dataset = pd.ewma(p_dataset, com = params["com"])
    return f_dataset
    
def ZLEMA(dataset, params):
    """cf filters"""
    p_dataset = to_absolute(dataset)
    
def KCA(dataset):
    """cf filters"""
    p_dataset = to_absolute(dataset)
    
# Predictive analysis
def predictions(f, params, dataset):
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
    Transfrom a price relatives dataframe to a price sequence dataframe
    The sequence if a pandas.series
    """
    return price_relatives.cumprod(axis=0)
