#!/usr/bin/env python
# coding: utf-8

# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups


# In[13]:


newsdata=fetch_20newsgroups()
newsdata


# In[14]:


newsdata.target_names


# In[15]:


categories=['talk.religion.misc','soc.religion.christian','sci.space','comp.graphics']
train_data=fetch_20newsgroups(subset='train', categories=categories)
test_data=fetch_20newsgroups(subset='test', categories=categories)


# In[16]:


train_data.target_names


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model_mb=make_pipeline(TfidfVectorizer(),MultinomialNB())


# In[18]:


model_mb.fit(train_data.data,train_data.target)


# In[19]:


labels=model_mb.predict(test_data.data)


# In[20]:


from sklearn.metrics import confusion_matrix
matr=confusion_matrix(test_data.target,labels)
sns.heatmap(matr.T,xticklabels=train_data.target_names, yticklabels=train_data.target_names, annot=True, fmt='d')


# In[21]:


def pred_doctype(s):
    pred=model_mb.predict([s])
    return train_data.target_names[pred[0]]


# In[22]:


pred_doctype('what is my screen resolution')


# In[23]:


pred_doctype('send this to international space station')


# In[24]:


pred_doctype('Study Hinduism vs atheism')


# In[ ]:




