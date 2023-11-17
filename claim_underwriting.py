#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew, stats
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,f1_score


# ## Data Import

# In[2]:


data = pd.read_csv("datasets_26475_38092_insurance2.csv")
data


# ## Exploratory Data Analysis

# ### Gender counts

# In[3]:


female = int (data.sex.value_counts()[0])
print (data.sex.value_counts())
print ("Prob Male: %.3f,\tProb Female: %.3f" % (1-(female/1338), (female/1338)))


# ### Smoker statistics

# In[4]:


non_smoker = data.smoker.value_counts()[0]
print (data.smoker.value_counts())
print ("Prob Non-smoker: %.3f,\tProb smoker: %.3f" % ((non_smoker/1338), 1-(non_smoker/1338)))


# ### BMI data statistics

# In[5]:


underweight = data.bmi.between(0,18.499999, inclusive=True).value_counts()[True]
normal = data.bmi.between(18.5, 24.999, inclusive=True).value_counts()[True]
overweight = data.bmi.between(25, 29.99999, inclusive=True).value_counts()[True]
obese = data.bmi.between(30, 10000, inclusive=True).value_counts()[True]
# print (underweight, normal, overweight, obese)
print ("underweight: \t%d,\t%.3f" % (underweight, underweight/1338))
print ("normal: \t%d,\t%.3f" % (normal, normal/1338))
print ("overweight: \t%d,\t%.3f" % (overweight, overweight/1338))
print ("obese: \t\t%d,\t%.3f" % (obese, obese/1338))


# ### Age stats

# In[6]:


underage = data.age.between(0,17.999, inclusive=True).value_counts().get(True, 0)
adult = data.age.between(18, 39.999, inclusive=True).value_counts().get(True, 0)
overadult = data.age.between(40, 59.999, inclusive=True).value_counts().get(True, 0)
old = data.age.between(60, 200, inclusive=True).value_counts().get(True, 0)
print ("0-18: \t%d,\t%.3f" % (underage, underage/1338))
print ("18-40: \t%d,\t%.3f" % (adult, adult/1338))
print ("40-60: \t%d,\t%.3f" % (overadult, overadult/1338))
print ("60-: \t%d,\t%.3f" % (old, old/1338))


# ### Summary Statistics

# In[7]:


data.describe()


# In[8]:


print("skew:  {}".format(skew(data)))


# In[9]:


print("kurtosis:  {}".format(kurtosis(data)))


# In[10]:


print("missing data values: \n{} ".format(data.isnull().sum()))


# In[11]:


sns.distplot(data['bmi'])


# In[12]:


sns.distplot(data['age'])  # Distribution of age


# In[13]:


sns.distplot(data['charges'])  # Distribution of charges


# In[14]:


sns.pairplot(data)


# ## Data Preparation

# In[15]:


X = data.drop(columns=['insuranceclaim'])
Y = data['insuranceclaim']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# In[17]:


# for comapritive analysis
score = pd.DataFrame()


# ## XGBoost

# In[18]:


model = XGBClassifier()


# In[19]:


model.fit(X_train, y_train)


# In[20]:


# make predictions for test data
y_pred = model.predict(X_test)


# In[21]:


# evaluate predictions
acc= np.round(accuracy_score(y_test, y_pred)*100,2)
acc


# In[22]:


f1= np.round(f1_score(y_test, y_pred)*100,2)


# In[23]:


print(classification_report(y_test, y_pred))


# In[24]:


score= score.append([["XGBoost", acc, f1]])


# ## Decision Trees

# In[25]:


dt = DecisionTreeClassifier(max_depth=1, random_state=28)
dt.fit(X_train, y_train)


# In[26]:


y_pred= dt.predict(X_test)


# In[27]:


acc= np.round(accuracy_score(y_pred,y_test)*100,2)
acc


# In[28]:


f1= np.round(f1_score( y_pred, y_test,average="weighted")*100,2)
f1


# In[29]:


print(classification_report(y_test, y_pred))


# In[30]:


score=score.append([['Decision Tree', acc, f1]])
score


# ## Support Vector Machines (SVMs)

# In[31]:


# SVM model intialization and generating pipeline for scaling variable for svm model
svm = Pipeline([('scaler', StandardScaler()),('svc', SVC(kernel='linear'))])


# In[32]:


svm.fit(X_train, y_train)


# In[33]:


y_pred = svm.predict(X_test)


# In[34]:


acc= np.round(accuracy_score(y_pred,y_test)*100,2)
acc


# In[35]:


f1= np.round(f1_score(y_test, y_pred, average="weighted")*100,2)
f1


# In[36]:


score=score.append([['SVM', acc, f1]])
score


# ## Naive Bayes

# In[37]:


gnb = GaussianNB()


# In[38]:


# fit
gnb.fit(X_train, y_train)


# In[39]:


# predict
y_pred_2 = gnb.predict(X_test)


# In[40]:


acc= np.round(accuracy_score(y_pred,y_test)*100,2)
acc


# In[41]:


f1= np.round(f1_score(y_test, y_pred, average="weighted")*100,2)
f1


# In[42]:


score=score.append([['Naive', acc, f1]])
score


# ## Comparision

# In[43]:


score.columns=['Model',"Accuracy",'F1-Score']


# In[44]:


score


# In[49]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(score['Model'],score['Accuracy'], color = 'b', width = 0.25)
plt.show()


# In[46]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(score['Model'],score['F1-Score'], color = 'b', width = 0.25)
plt.show()

