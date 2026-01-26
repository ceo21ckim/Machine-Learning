import pandas as pd, numpy as np
from typing import Union
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dataframe:pd.DataFrame, feature_names:Union[list, np.array], target_name):
        self.dataframe = dataframe 
        self.feature_names = feature_names
        self.target_name = target_name

        self.X = self.dataframe[self.feature_names]
        self.y = self.dataframe[self.target_name]


    def train_test_split(self, **kwargs):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, **kwargs)
        print(f'X_train shape: {self.X_train.shape}\ty_train shape: {self.y_train.shape}')
        print(f'X_test shpae: {self.X_test.shape}\ty_test shape: {self.y_test.shape}')
        return self.X_train, self.X_test, self.y_train, self.y_test    
    
    def preprocessing(self, X):
        raise NotImplementedError
