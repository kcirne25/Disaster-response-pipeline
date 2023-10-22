#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[20]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 200)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


# In[4]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)
df


# In[5]:


# Descriptive statistics to verify dataset
df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


#Removing child_alone  column as it is not meaningful, the column has just 0 on its values
df = df.drop(['child_alone'],axis=1)


# The column 'Related' has a max value of '2', which could be an error as this column looks like a column with binominal values

# In[9]:


# Checking count of '2' on related column
df.groupby('related').count()


# In[10]:


# Replacing value '2' with '1' on related column
df['related'] = df['related'].map(lambda x: 1 if x==2 else x)
df.groupby('related').count()


# In[11]:


X = df['message']
y = df.iloc[:,4:]


# ### 2. Write a tokenization function to process your text data

# In[16]:


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


# In[17]:


# Creating custom transformer to be used in the ML pipeline

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[18]:


def ML_pipeline(clf = AdaBoostClassifier()):
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    return pipeline

# include StartingVerbExtractor custom transformer
def ML_pipeline_2(clf = AdaBoostClassifier()):
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    return pipeline


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[21]:


# Splitting data and training pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = ML_pipeline()
model.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[22]:


# Testing model and verifying results

y_pred_test = model.predict(X_test)

# classification report on test data
print(classification_report(y_test.values, y_pred_test, target_names=y.columns.values))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[23]:


# get list of model parameters
model.get_params().keys()


# In[24]:


model_2 = ML_pipeline()

# Model training using GridSearchCV is computational extensive task. 
# Computational time increases as we increase the number of parameters.
# In view of that, only a few parameters are chosen to demonstrate the concept.
# To obtain a highly optimized model, we need to increase the number of parameters.

parameters = {
    'clf__estimator__learning_rate': [0.5, 1.0],
    'clf__estimator__n_estimators': [10, 20]
}


cv = GridSearchCV(model_2, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
# verbose=3 to get real time training progress
# n_jobs=-1 -> to train in parallel across the maximum number of cores in our computer, spending it up.
# cv=5 -> 5-fold cross validation


cv.fit(X_train, y_train)


# In[25]:


model_2 = ML_pipeline()

# Model training using GridSearchCV is computational extensive task. 
# Computational time increases as we increase the number of parameters.
# In view of that, only a few parameters are chosen to demonstrate the concept.
# To obtain a highly optimized model, we need to increase the number of parameters.

parameters = {
    'clf__estimator__learning_rate': [0.5, 1.0],
    'clf__estimator__n_estimators': [10, 20]
}


cv = GridSearchCV(model_2, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
# verbose=3 to get real time training progress
# n_jobs=-1 -> to train in parallel across the maximum number of cores in our computer.
# cv=5 -> 5-fold cross validation


cv.fit(X_train, y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[26]:


y_pred_test = cv.predict(X_test)

# classification report
print(classification_report(y_test.values, y_pred_test, target_names=y.columns.values))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[27]:


# trying RandomForestClassifier model 
rf_model = ML_pipeline(clf = RandomForestClassifier())
rf_model.fit(X_train, y_train)
y_pred_rf_test = rf_model.predict(X_test)

print(classification_report(y_test.values, y_pred_rf_test, target_names=y.columns.values))


# In[28]:


# Trying ML_pipeline_2 which includes custom transformer of 'StartingVerbEstimator'
model_3 = ML_pipeline_2()
model_3.fit(X_train, y_train)
y_pred_3_test = model_3.predict(X_test)

print(classification_report(y_test.values, y_pred_3_test, target_names=y.columns.values))


# ### 9. Export your model as a pickle file

# In[29]:


# save model in pickle file
with open('classifier.pkl', 'wb') as f:
    pickle.dump(model_3, f)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




