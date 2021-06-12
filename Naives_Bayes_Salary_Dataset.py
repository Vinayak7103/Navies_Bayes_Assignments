#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


salary_train = pd.read_csv("C:/Users/vinay/Downloads/SalaryData_Train.csv")
salary_test = pd.read_csv("C:/Users/vinay/Downloads/SalaryData_Test.csv")


# In[4]:


salary_test


# In[5]:


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


# ## Preprocessing the data. As, there are categorical variables

# In[6]:


from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()


# In[7]:


for i in string_columns:
    salary_train[i]= number.fit_transform(salary_train[i])
    salary_test[i]=number.fit_transform(salary_test[i])


# ## Capturing the column names which can help in futher process

# In[12]:


colnames = salary_train.columns
colnames


# In[9]:


len(colnames)


# In[10]:


x_train = salary_train[colnames[0:13]]
y_train = salary_train[colnames[13]]
x_test = salary_test[colnames[0:13]]
y_test = salary_test[colnames[13]]


# In[11]:


x_test


# ## Building Multinomial Naive Bayes model

# In[14]:


from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


# In[16]:


classifier_mb = MB()
classifier_mb.fit(x_train,y_train)


# In[18]:


pred_mb = classifier_mb.predict(x_train)
accuracy_mb_train = np.mean(pred_mb == y_train)


# In[19]:


accuracy_mb_train ##77%


# In[20]:


pd.crosstab(pred_mb, y_train)


# ## for test data

# In[21]:


pred_mb_test = classifier_mb.predict(x_test)
accuracy_mb_test = np.mean(pred_mb_test == y_test)


# In[22]:


accuracy_mb_test  #77%


# In[23]:


pd.crosstab(pred_mb_test, y_test)


# ## Building Gaussian model

# In[24]:


classifier_gb = GB()
classifier_gb.fit(x_train, y_train)
pred_gb = classifier_gb.predict(x_train)
accuracy_gb_train = np.mean(pred_gb == y_train)


# In[25]:


accuracy_gb_train #80%


# In[26]:


pd.crosstab(pred_gb,y_train)


# In[27]:


##for test data
pred_gb_train = classifier_gb.predict(x_test)
accuracy_gb_test = np.mean(pred_gb_train == y_test)


# In[28]:


accuracy_gb_test ##80%


# In[29]:


pd.crosstab(pred_gb_train,y_test)


# In[ ]:




