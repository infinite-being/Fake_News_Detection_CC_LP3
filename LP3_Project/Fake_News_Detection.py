#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries.

# In[ ]:


import  pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import sklearn


# Loading data set into raw_data variable.

# In[38]:


raw_data=pd.read_csv('Desktop/news.csv')


# Displaying the content of the data set 'news.csv'.

# In[39]:


raw_data


# Displaying the shape of the data set.

# In[4]:


raw_data.shape


# Describing the data set like its count, mean, min, max, etc.

# In[5]:


raw_data.describe()


# Displaying only the top 5 row using .head command.

# In[6]:


raw_data.head()


# Displaying the last 5 rows using .tail.

# In[7]:


raw_data.tail()


# Dropping the field 'Unnamed:0' as its irrelevant index field.

# In[8]:


data=raw_data.drop(['Unnamed: 0'],axis=1)


# In[9]:


data


# Declaring a variable y with 'label' field, on the basis of y we will build our model. As the field y depicts whether a news is fake or real.

# In[10]:


y=data['label']
y


# Dropping the column 'label' as it is no longer required in data variable as we have saved 'label' as 'y' variable.

# In[11]:


data.drop(['label'],axis=1)


# Importing relevant libraries from sklearn. Spliting the data sets into 'train' and test' helps us to split our data set into two sets in a specific ratio, so that we can train our model on the basis of 'train' set and then apply the model on 'test' set to predict the accuracy of our model we have built. Making two unique sets each time we run the model is assured by giving it a random state value. So everytime the model is run, 'train' and 'test' sets are mutually exclusive to each other every time also the data within them is randomly arrange but the data between two sets are never mixed, no matter how many times we run the model. Random state takes care of that.
# Usually its recmmended that the split is 80-20, 80%of original data set is given to train set and 20% to test data set. But here we have make it 3:1, to build more robust model.

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.33, random_state=53)


# In[14]:


x_train


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[17]:


x_test


# In[18]:


y_train.shape


# In[19]:


y_train


# In[20]:


y_test.shape


# In[21]:


y_test


# Importing relevant libraries from sklearn ,'TfidfVectorizer' by using it we are fiting our model's train set and transforming into 'tfidf_train' set. Similarly fiting test data into 'tfidf_test'.

# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[23]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[24]:


tfidf_train = tfidf_vectorizer.fit_transform(x_train) 


# In[25]:


tfidf_train


# In[26]:


tfidf_test = tfidf_vectorizer.transform(x_test)


# In[27]:


tfidf_test


# In[28]:


print(tfidf_vectorizer.get_feature_names()[-10:])


# Importing relevant libraries from sklearn so that we can use 'Passive Aggressive Classifier', so that we can finally test our model on 'TfidfVectorizer' and at the end calling the confusion matrix and accuracy_score to finally check how well our model is, by evaluating the confusion_matrix and the accuracy being predicted. 

# In[36]:


from sklearn.linear_model import PassiveAggressiveClassifier as pc
from sklearn.metrics import accuracy_score, confusion_matrix


# In[30]:


clf=pc()


# In[31]:


clf.fit(tfidf_train,y_train)


# In[40]:


pred = clf.predict(tfidf_test)
score =accuracy_score(y_test, pred)
score


#   Above is the Accuracy and Below is the confusion matrix of our model.

# In[42]:


confusion_matrix(y_test,pred)


# 
#     CONCLUSION: 
#    
# Accuracy of our model is 93.78%, means approx 94% of the time the model predicted the news fake/real. Which we can consider it to be a successful model but needs more variation of methods. Stll, it was able to predict 93.78% of the time based on our train data set tested on our test data set.
#     
#     Confusion matrix shows us: 
# 1) 952 times it predicted 'FAKE' news while it was a 'FAKE' news, though it predicted wrong 56 times and predicted 'REAL' although the news was 'FAKE'.
# 
# 2) 1004 times it predicted 'REAL' news while it was a 'REAL' news, though it predicted wrong 79 times and predicted 'FAKE' although the news was 'REAL'.
# 
# 
# 
