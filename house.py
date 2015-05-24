'''
house.py - Model for predicting the selling 
           price of a house from other attributes.

Author: Phillip Schafer (phillip.baker.schafer@mg.thedataincubator.com)
Date: May 8, 2015

Usage:
>> python house.py

Dependencies: transformers.py
'''
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import feature_extraction
from sklearn import neighbors
import transformers

def load_df(file, price_needed=True):
    ''' Load csv file as a pandas dataframe and clean it up '''
    df = pd.read_csv(file)
    if price_needed:
        # remove rows with no ClosePrice
        df = df[df['ClosePrice'].notnull()]
    # There is one data point with living area 9999999 instead of NaN... replace.
    df['LivingArea'].replace(9999999, np.NaN, inplace=True)
    # replace NaN's in PublicRemarks with empty strings
    df['PublicRemarks'].replace(np.NaN, '', inplace=True)
    
    return df

def build_pipeline(num_vars, cat_vars, geo_vars, txt_vars):
    '''
    Build a model pipeline for predicting the house prices from various 
    types of data
    '''
    # numerical
    num_pl = pipeline.Pipeline([
        # select numerical data
        ('number_selector', transformers.ColumnSelectTransformer(num_vars)), 
        # impute missing values (some LivingArea values are NaN)
        ('imputer', preprocessing.Imputer(strategy='median')),
        # normalize
        ('normalizer', preprocessing.StandardScaler()),
    ])

    # categorical
    cat_pl = pipeline.Pipeline([
        # select numerical data
        ('number_selector', transformers.ColumnSelectTransformer(cat_vars)), 
        # convert dataframe to dictionary 
        ('dict_maker',  transformers.DictTransformer()),
        # vectorize
        ('vectorizer', feature_extraction.DictVectorizer()),
    ])
    
    # geographical
    geo_pl = pipeline.Pipeline([
        # select lat/long data
        ('loc_selector', transformers.ColumnSelectTransformer(geo_vars)), 
        # impute missing values
        ('imputer', preprocessing.Imputer(strategy='median')),
        # model
        ('regressor', neighbors.KNeighborsRegressor()),
    ])

    # unstructured text
    txt_pl = pipeline.Pipeline([
        # select text field
        ('text_selector', transformers.ColumnSelectTransformer(txt_vars)), 
        # convert to list of strings
        ('text_transformer', transformers.ListTransformer()),
        # extract TF-IDF normalized featueres
        ('vectorizer', feature_extraction.text.TfidfVectorizer(
            stop_words='english', min_df=.0001, max_df=.5)),
    ])

    # full model pipeline
    pl = pipeline.Pipeline([
        ('union', pipeline.FeatureUnion(
            transformer_list=[
                ('num_pl', num_pl),
                ('cat_pl', cat_pl),
                ('geo_pl', transformers.EstimationTransformer(geo_pl)),
                ('txt_pl', txt_pl)
            ]
        )),
        ('regressor', linear_model.LinearRegression())
    ])
    
    return pl


class PricePredict(object):
    ''' Class for modeling the house prices '''
    def __init__(self, file, use_listprice=False):
        ''' Load the training data and initialize the model '''
        self.df = load_df(file, price_needed=True)
        self.prices = self.df[['ClosePrice']].values
        
        self.num_vars = ['LivingArea', 'NumBedrooms', 'NumBaths']
        self.cat_vars = ['Pool', 'DwellingType', 'ExteriorStories']
        self.geo_vars = ['GeoLat', 'GeoLon']
        self.txt_vars = ['PublicRemarks']
        if use_listprice:
            self.num_vars.append('ListPrice')
        
        self.pl = build_pipeline(self.num_vars, self.cat_vars, 
                                 self.geo_vars, self.txt_vars)
        
    def validate(self):
        '''
        Validate the model using 5-fold cross validation on the training set
        '''
        scores = cross_validation.cross_val_score(
            self.pl, self.df, self.prices, cv=5)
        return scores.mean(), scores.std()
    
    def fit(self):
        ''' Fit the model to the full training set '''
        self.pl.fit(self.df, self.prices)
        
    def predict(self, file):
        ''' Make predictions for test data '''
        test_df = load_df(file, price_needed=False)
        return self.pl.predict(test_df)

def main():
    file = 'data_sci_snippet.csv'
    
    print "Training model with 'ListPrice' included..."
    model = PricePredict(file, use_listprice=True)
    mn, std = model.validate()
    print("Accuracy: %0.2f (+/- %0.2f)\n" % (mn, std * 2))    
    
    print "Training model without 'ListPrice' included..."
    model = PricePredict(file, use_listprice=False)
    mn, std = model.validate()
    print("Accuracy: %0.2f (+/- %0.2f)\n" % (mn, std * 2))
    
    #test_file=file
    #model.fit()
    #print model.predict(test_file)[:10]
    
if __name__ == "__main__":
    main()

