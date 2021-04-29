#!/usr/bin/env python
# coding: utf-8

# In[19]:


import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split #to split to test and train set
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns


# In[20]:


from sklearn.naive_bayes import GaussianNB


# In[21]:


df=pd.read_csv("/Users/manikhossain/Downloads/SOEN-6111_BigData_Materials/SOEN6111_project/dataset/iot_telemetry_data.csv")


# In[22]:


# investigation for duplicates
df.shape
#(405184, 9)

df.drop_duplicates().shape
#405171 rows × 9 columns

# same values were recorded twice => we will drop these 14 duplicates 
df[df.duplicated(keep=False) == True]


# In[23]:


# drop the duplicates and keep the first occurence only, modify the dataframe
df.drop_duplicates(inplace=True, keep="first")
# verification that there are no duplicates
df[df.duplicated(keep=False) == True]


# In[24]:


# there are no NULL values
df.isna().sum()


# In[25]:


# how many data points we have per device
df.groupby("device").size()


# In[26]:


# convert unix epochs to datetime
df["measure_time"]=pd.to_datetime(df["ts"], unit = 's')
df.head(10)


# In[27]:


#df["ts"].iloc[0]
#1594512094.3859746
#take the time up to seconds precision
# convert unix epochs to datetime
df["measure_seconds"]=pd.to_datetime(df["ts"].astype(int), unit = 's')

#how many data points we have per sencond
df_entries_sec=df.groupby("measure_seconds").size().reset_index(name="count")

# For an interval of second we can have up to 3 entries
df_entries_sec["count"].max()


# In[28]:


# what is the proportion of the frequency per second
df_entries_sec.loc[(df_entries_sec['count'] == 3)].count()
#4651 times we receive 3 data entries per second

df_entries_sec.loc[(df_entries_sec['count'] == 2)].count()
#62246 times we received 2 data entries per second

df_entries_sec.loc[(df_entries_sec['count'] == 1)].count()
#266726 -> Most of the data entries are sent at frequency 1 per second


# In[29]:


# do we have long periods without any entries?
import datetime
start_dte = datetime.datetime(2020, 7, 12)
print(start_dte)

#find out the last date
df.loc[(df['measure_time'] >= "2020-07-20")]

end_dte = datetime.datetime(2020, 7, 20, 0, 3, 27)
print(end_dte)

# create date range per second inclusive of the end date
ref_date_range = pd.date_range(start=start_dte, end=end_dte, freq='S')
print(len(ref_date_range))
#691408 seconds in total

missing_seconds =~ref_date_range.isin(df.measure_seconds)
#numpy.ndarray

# approximately half of the times there is no entry
unique, counts = np.unique(missing_seconds, return_counts=True)
dict(zip(unique, counts))
#{False: 333618, True: 357790}


# In[30]:


#select the features and the target variable
#we exclude the timestamp
#?Question? Do we need to add the device?
feature_cols = ['co', 'humidity', 'lpg', 'motion','smoke','temp']
X = df[feature_cols] # Features
y = df['light'] # Target variable


# # Train & Test Split() : train=80% & test=20%

# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# # Over-sampling using SMOTE

# In[37]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# In[41]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# # Gaussian Naive Bayes

# In[44]:


classifierGNB = GaussianNB()
classifierGNB.fit(X_train_res, y_train_res)
y_pred = classifierGNB.predict(X_test)
 
# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[45]:


print(' Normalized Confusion Matrix: Gaussian Naive Bayes   ')
confusion_matrixGNB = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize=True)
sns.heatmap(confusion_matrixGNB, annot=True, )
plt.show()


# # Multinomial Naive Bayes

# In[46]:


from sklearn.naive_bayes import MultinomialNB

classifierMNB = MultinomialNB()
classifierMNB.fit(X_train_res, y_train_res)
y_pred = classifierMNB.predict(X_test)


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[47]:


confusion_matrixMNB = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize=True)
sns.heatmap(confusion_matrixMNB, annot=True, )
plt.show()


# In[49]:


from sklearn.metrics import accuracy_score, log_loss, roc_curve
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB


classifiers = [
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    ComplementNB(),]
 
# Logging for Visual Comparison
log_cols=["Naive Bayes Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
 
for clf in classifiers:
    clf.fit(X_train_res, y_train_res)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, train_predictions))
    log_entry = pd.DataFrame([[name, acc*100, 11]], columns=log_cols)
    log = log.append(log_entry)
    
    print("="*30)

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Naive Bayes Classifier', data=log, color="b")
 
plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()


# In[ ]:




