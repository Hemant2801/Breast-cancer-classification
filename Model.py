#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Data collection and analysis

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Breast cancer classification/data.csv')


# In[3]:


# Print the first 5 rows of the dataset
df.head()


# In[4]:


df = df.drop('Unnamed: 32', axis =1)


# In[5]:


# Shape of the dataset
df.shape


# In[6]:


# Getting some info about the dataset
df.info()


# In[7]:


# Checking for any null values
df.isnull().sum()


# In[8]:


df.head()


# In[9]:


# Stastical measure about the dataset
df.describe()


# In[10]:


# Checking the distribution of taget variable
df['diagnosis'].value_counts()


# In[11]:


# Converting the categorical data into numerical data
# B = 0 --> Benign
# M = 1 --> Malignant
df.replace({'diagnosis' : {'B' : 0, 'M' : 1}}, inplace = True)


# In[12]:


df['diagnosis'].value_counts()


# In[13]:


df.groupby('diagnosis').mean()


# In[14]:


plt.figure(figsize = (6,6))
sns.countplot(x = 'diagnosis', data = df)
plt.show()


# Separating the feature and target

# In[15]:


X = df.drop(columns = ['id', 'diagnosis'], axis = 1)
Y = df['diagnosis']


# In[16]:


print(X)
print(Y)


# Split the data into training and testing data

# In[17]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size =.2, stratify = Y, random_state = 2)


# In[18]:


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# # Model training
# 
# logistic regression

# In[19]:


model = LogisticRegression()


# In[20]:


model.fit(x_train, y_train)


# Model evaluation:
# 
# accuracy score

# In[21]:


# on training data
training_predict = model.predict(x_train)

training_accuracy = accuracy_score(y_train, training_predict)
print('TRAINING ACCURACY :', training_accuracy)


# In[22]:


# on testing data
testing_predict = model.predict(x_test)

testing_accuracy = accuracy_score(y_test, testing_predict)
print('TESTING ACCURACY :', testing_accuracy)


# # Building a predictive system

# In[23]:


input_data = input()

list_data = [float(i) for i in input_data.split(',')]

array_data = np.asarray(list_data)

reshaped_array = array_data.reshape(1, -1)

prediction = model.predict(reshaped_array)
if prediction == 0:
    print('THE CANCER IS BENIGN')
else:
    print("THE CANCER IS MALIGNANT")


# In[ ]:




