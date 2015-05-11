'''
transformers.py - Transformer class utilities for use with `scikit-learn`

Author: Phillip Schafer (phillip.baker.schafer@mg.thedataincubator.com)
Date: May 8, 2015
'''
from sklearn import base
import pandas as pd
import numpy as np

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    '''
    Transformer class that selects columns from a pandas dataframe to a numpy 
    array of floats
    '''
    def __init__(self, cols):
        ''' Constructor takes `cols`, a list of column labels to use '''
        self.cols = cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        ''' X should be a pandas DataFrame. Output selected columns. '''
        return X.loc[:, self.cols]
    
class DictTransformer(base.BaseEstimator, base.TransformerMixin): 
    ''' 
    Transformer class that converts a pandas dataframe into a dict 
    representation that can be fed into DictVectorizer
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        ''' X should be a pandas DataFrame. Output a list of dicts. '''
        return [{column:val for column, val in row.iteritems()} for _, row in X.iterrows()] 
    
class EstimationTransformer(base.BaseEstimator, base.TransformerMixin):
    '''
    Transformer class that converts an estimator into a transformer
    '''
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, x, y=None):
        self.estimator.fit(x, y)
        return self
        
    def transform(self, X):
        return self.estimator.predict(X)
    
    
class ListTransformer(base.BaseEstimator, base.TransformerMixin):
    ''' 
    Transformer that converts a pandas dataframe into a list
    '''
    def fit(self, x, y=None):
        return self
        
    def transform(self, X):
        return X.values.ravel()