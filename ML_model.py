
# coding: utf-8

# In[49]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as pt


#importing dataset

data  =  pd.read_csv('covid-19 in india.csv')

x_data=data.iloc[:,1:2].values

y_data=data.iloc[:,2:3].values

#scaling the data

'''from sklearn.preprocessing import StandardScaler

scale=StandardScaler()
x_data=scale.fit_transform(x_data)

y_data=scale.fit_transform(y_data)'''

# spliting data 

from sklearn.cross_validation import train_test_split

x_traning,x_test,y_training,y_test=train_test_split(x_data,y_data,test_size=0.50,random_state=0)



# training model

from sklearn.linear_model import LinearRegression 

regressor=LinearRegression()

regressor.fit(x_traning,y_training)

t=regressor.predict(x_test)

#saving object 

from sklearn.externals import joblib
joblib.dump(regressor,'corona_model.pkl')

m=joblib.load('corona_model.pkl')


# In[41]:




