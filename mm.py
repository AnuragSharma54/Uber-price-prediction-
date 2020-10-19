import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Dataset 
data = pd.read_csv('taxi.csv')
data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3,random_state=0)