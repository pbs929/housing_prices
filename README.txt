GOAL
----

The aim of this project is to create a model that predicts the selling price of a house based upon other attributes.  A sample input file is 'data_sci_snippet.csv'.

CODE STRUCTURE
--------------

The file 'house.py' contains the class `PricePredict`, which builds and tests the model.  
-> The class constructor takes the name of the file containing the training data as an argument, and loads the data
-> The validate() method evaluates the model using 5-fold cross-validation on the training data.  It returns an array of r^2 values from the 5 trials.  
-> The fit() method fits the model using the full training set (not used here)
-> The predict() method applies the model on testing data (not used here)

The file also contains two public functions used by the `PricePredict` class:
-> `load_df` loads the data into a pandas dataframe and cleans it up
-> `build_pipeline` returns the model pipeline as described in this README

The `main()` function builds and validates a model using the given sample data.   

Finally, an additional file `transformers.py` contains simple transformer classes for use in building pipelines.  

APPROACH
--------

The problem was to predict home prices given other listing information.  To begin with, I broke down the various types of data available:

** Numerical **
LivingArea
NumBedrooms 
NumBaths 
ListPrice

** Categorical **
Pool 
DwellingType 
ExteriorStories (treated as categorical because typically only 1 or 2)

** Unstructured **
PublicRemarks

** Other **)
GeoLat, GeoLon
ListDate, CloseDate (disregarded here)

My general strategy was to first build a simple model and then add to it by incorporating more types of data.  I evaluated the models' performance using an r^2 metric with 5-fold cross-validation.  All model building and evaluation was done using python's scikit-learn package.  The data is represented in a pandas dataframe.  

I began by simply training a linear regression model on the numerical data alone.  I built a pipeline that selects the numerical columns in the dataframe, imputes missing values, and mean-variance scales the data.  Regression gave an r^2 score of 0.95 when 'ListPrice' was included, and 0.56 when 'ListPrice' was excluded.  Because the performance with the list prices included was already good, I evaluated the following model improvements only for the excluded case.  

I next included the categorical variables by building a pipeline that converts them to numerical variables using one-hot encoding (using scikit-learn's DictVectorizer).  These variables on their own were poor predictors of price (r^2=0.15 using a cross-validated ridge regression) but when combined with the numerical variables improved the linear regression score to r^2=0.58.  

Next, I worked on incorporating the geographical (latitude/longitude) data.  I built a pipeline that predicts the price based on these variables alone, using k-nearest neighbors regression.  (This predicts the price based on prices of nearby homes.)  The model by itself had an R^2 score of 0.62.  To incorporate this model's output into the larger model, I converted the geographically-predicted price into an additional feature for linear regression.  The combined model now had a score of r^2=0.73.  

Finally, I added in the unstructured text data using a bag-of-words model.  I extracted word counts and TF-IDF normalized them using scikit-learn's TfidfVectorizer.  The text features alone gave a score of r^2=0.52.  (I found the optimal values min_df=.0001, max_df=.5 through a grid search.)  When incorporated into the larger model, the text features improved the score to r^2=0.79

FURTHER DIRECTIONS
------------------

I gained significant improvements by incorporating the various types of data (from 0.56 using only numerical data to 0.79 in the end) using linear regression.  Other, more complex models could be built.  For one thing, since the unstructured text features are sparse, a regularized linear model would probably be preferable, although by playing with this a little I got only very small improvements.  A random forest or other nonlinear model might also be able to do even better.  We could also use the geographical data to pull in other data about the area each house is in, and could introduce more sophisticated ways of extracting features from the unstructured text.  