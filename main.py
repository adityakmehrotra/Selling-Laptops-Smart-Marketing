import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import sqlite3
import rasterio
import sklearn
import copy
import sklearn.linear_model

from rasterio.mask import mask
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

class UserPredictor:
    def fit(self, train_users, train_logs, y_pred):
        # Create a new dataframe that is all of the important information from the _users, _logs, and _y combined
        main_df = {}
        main_df = pd.merge(train_users, y_pred, on = 'user_id', how = 'outer')
        main_df = pd.merge(main_df, train_logs, on = 'user_id', how = 'outer')
        
        
        print(main_df)
        # Create an id Dictionary tracking when the user_id changes to keep track of how many users there are
        idDictionary = {}
        for index, row in main_df.iterrows():
            userIDVar = row['user_id']
            if userIDVar not in idDictionary:
                
                idDictionary[userIDVar] = 1
            else:
                idDictionary[userIDVar] += 1
        
        temporaryDictionary = {}
        
        temporaryDictionary = pd.DataFrame.from_dict(idDictionary, orient = 'index')
        temporaryDictionary = temporaryDictionary.set_index(temporaryDictionary.index)
        
        temporaryDictionary = temporaryDictionary.set_index(temporaryDictionary.index)
        
        
        temporaryDictionary = temporaryDictionary.rename_axis('user_id')
        
        temporaryDictionary = temporaryDictionary.rename(columns = {0: 'total_visits'})
        
        print(temporaryDictionary)
        
        # Merge the temporary Dictionary created above along with the combined DataFrame
        fullData = {}
        fullData = pd.merge(main_df, temporaryDictionary, on = 'user_id', how = 'outer')
        
        
        # Add a new coloumn called Coef, and set all of the values to 1
        fullData['coef'] = 1
    
        # Remove all of the duplicates in the combined DataFrame in the user_id column
        fullData = fullData.drop_duplicates(subset = ['user_id'])
        print(fullData)
        
        # Columns the model will use
        xcols = ['age', 'badge', 'coef', 'past_purchase_amt', 'total_visits']
        ycol = 'y'
        
        
        # Transform the data by using OneHotEncoder on the Age and Badge column along with StandarScalar
        # on Total Visits, Past Purchase Amount, and the Coefficient column
        customTransformation = make_column_transformer (
            (OneHotEncoder(), ['age', 'badge']),
            (StandardScaler(), ['coef', 'past_purchase_amt', 'total_visits']),
        )
        customTransformation
        
        # Actually create the m1. It is LogisticRegression.
        
        self.m1 = Pipeline([
            ('transformers', customTransformation),
            ('lr', LogisticRegression(fit_intercept = False, max_iter = 1000000)),
        ])
        
        # Split the Data into train, and test, then fit the data
        train, test = train_test_split(fullData)
        print(fullData)
        self.m1.fit(train[xcols], train[ycol])
        
    def predict(self, test_users, test_logs):
        # Create a new dataframe that is all of the important information from the _users, and _logs
        main_df = {}
        main_df = pd.merge(test_users, test_logs, on = 'user_id', how = 'outer')
        print(main_df)
        
        # Create an id Dictionary tracking when the user_id changes to keep track of how many users there are
        idDictionary = {}
        
        for value, row in main_df.iterrows():
            userIDVar = row['user_id']
            if userIDVar not in idDictionary:
                idDictionary[userIDVar] = 1
                
            else:
                idDictionary[userIDVar] += 1
        print(idDictionary)
        temporaryDictionary = {}
        temporaryDictionary = pd.DataFrame.from_dict(idDictionary, orient = 'index')
        temporaryDictionary = temporaryDictionary.set_index(temporaryDictionary.index)
        
        temporaryDictionary = temporaryDictionary.rename_axis('user_id')
        
        temporaryDictionary = temporaryDictionary.rename(columns = {0: 'total_visits'})
        
        print(temporaryDictionary)
        
        # Merge the temporary Dictionary created above along with the combined DataFrame
        fullData = {}
        
        
        fullData = pd.merge(main_df, temporaryDictionary, on = 'user_id', how = 'outer')
        
        # Add a new coloumn called Coef, and set all of the values to 1
        
        print("Before coefficient column")
        print(fullData)
        fullData['coef'] = 1
        print(fullData)
        print("After coefficient column")
        
        # Remove all of the duplicates in the combined DataFrame in the user_id column
        fullData = fullData.drop_duplicates(subset = ['user_id'])
        
        print(fullData)
        # Columns the model will use
        
        xcols = ['age', 'badge', 'coef', 'past_purchase_amt', 'total_visits']
        
        # Get the results and return it
        results = np.array(self.m1.predict(fullData[xcols]))
        
        
        return results
