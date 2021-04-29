#!/usr/bin/env python
# coding: utf-8

# In[196]:


import numpy as np
import pandas as pd


# In[197]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[198]:


from sklearn.model_selection import train_test_split #to split to test and train set
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# In[199]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns


# In[200]:


from sklearn.naive_bayes import GaussianNB


# In[201]:


df=pd.read_csv("/Users/manikhossain/Downloads/SOEN-6111_BigData_Materials/SOEN6111_project/dataset/iot_telemetry_data.csv")
# /Users/manikhossain/Downloads/SOEN-6111_BigData_Materials/SOEN6111_project/dataset/
#!pwd


# In[202]:


#explore the data frame
df.info()


# In[203]:


df.columns


# In[204]:


# investigation for duplicates
df.shape
#(405184, 9)

df.drop_duplicates().shape
#405171 rows × 9 columns

# same values were recorded twice => we will drop these 14 duplicates 
df[df.duplicated(keep=False) == True]


# In[205]:


# drop the duplicates and keep the first occurence only, modify the dataframe
df.drop_duplicates(inplace=True, keep="first")
# verification that there are no duplicates
df[df.duplicated(keep=False) == True]


# In[206]:


# there are no NULL values
df.isna().sum()


# In[207]:


# how many data points we have per device
df.groupby("device").size()


# In[208]:


# convert unix epochs to datetime
df["measure_time"]=pd.to_datetime(df["ts"], unit = 's')
df.head(10)


# In[209]:


#df["ts"].iloc[0]
#1594512094.3859746
#take the time up to seconds precision
# convert unix epochs to datetime
df["measure_seconds"]=pd.to_datetime(df["ts"].astype(int), unit = 's')

#how many data points we have per sencond
df_entries_sec=df.groupby("measure_seconds").size().reset_index(name="count")

# For an interval of second we can have up to 3 entries
df_entries_sec["count"].max()


# In[210]:


# what is the proportion of the frequency per second
df_entries_sec.loc[(df_entries_sec['count'] == 3)].count()
#4651 times we receive 3 data entries per second

df_entries_sec.loc[(df_entries_sec['count'] == 2)].count()
#62246 times we received 2 data entries per second

df_entries_sec.loc[(df_entries_sec['count'] == 1)].count()
#266726 -> Most of the data entries are sent at frequency 1 per second


# In[211]:


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


# In[212]:


# what is the longest sequence of False
max_period=0
max_period_frequency=0
current_count=0
for elem in missing_seconds:
    if elem:
        current_count +=1
    else:
        if (max_period < current_count):
            max_period = current_count
        if current_count == 2:
            max_period_frequency+=1
        current_count=0
print(max_period)
print(max_period_frequency)
#The longuest period where we dont have entries is 6 seconds 
#and this happened 4 times

# 4 times we had to wait 6 seconds for an entry to come in
# 142 times we had to wait 5 seconds for an entry to come in
# 2974 times we had to wait 4 seconds for an entry to come in
# 18096 times we had to wait 3 seconds for an entry to come in
# 54339 times we had to wait 2 seconds for an entry to come in
# 57981 times we had to wait 1 second for an entry to come in

# the longest period when we received consecutively entries is 94 seconds
# 41150 times we received data without interruption for 2 seconds
# 35815 times we received data consecutively for 3 seconds
# 2 times we received data consecutively for 4 seconds


# In[213]:


#summary of the distribution of continuous variables:
df.describe()


# In[214]:


#??? Why ligth and movement do not appear in describe?
#Should I convert their values to 1 and 0
df["light"] = df["light"].astype(int)
df["motion"] = df["motion"].astype(int)
df.describe()


# In[215]:


#??? Is the data set balanced? we have to count how many instances we have for each class
df['light'].value_counts()

#0    292649
#1    112522
# => the data set is not balanced


# # Train & Test Split() : train=80% & test=20%

# In[216]:


#select the features and the target variable
#we exclude the timestamp
#?Question? Do we need to add the device?
feature_cols = ['co', 'humidity', 'lpg', 'motion','smoke','temp']
X = df[feature_cols] # Features
y = df['light'] # Target variable


# In[217]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # GaussianNB()

# In[240]:


classifierGNB = GaussianNB()
classifierGNB.fit(X_train, y_train)
y_pred = classifierGNB.predict(X_test)
 
# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[247]:


print(' Normalized Confusion Matrix: Gaussian Naive Bayes   ')
confusion_matrixGNB = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize=True)
sns.heatmap(confusion_matrixGNB, annot=True, )
plt.show()


# # MultinomialNB()

# In[220]:


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[222]:


confusion_matrixMNB = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize=True)
sns.heatmap(confusion_matrixMNB, annot=True, )
plt.show()


# # BernoulliNB()

# In[223]:


# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
 
# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[224]:


confusion_matrixBNB = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize=True)
sns.heatmap(confusion_matrixBNB, annot=True, )
plt.show()


# # ComplementNB()

# In[225]:


# Complement Naive Bayes
from sklearn.naive_bayes import ComplementNB
classifier = ComplementNB()
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
 
# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[235]:


confusion_matrixCNB = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize=True)
sns.heatmap(confusion_matrixCNB, annot=True)
plt.show()


# In[239]:


from sklearn.metrics import accuracy_score, log_loss, roc_curve
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


classifiers = [
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    ComplementNB(),]
 
# Logging for Visual Comparison
log_cols=["Naive Bayes Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
 
for clf in classifiers:
    clf.fit(X_train, y_train)
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




