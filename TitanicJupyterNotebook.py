#!/usr/bin/env python
# coding: utf-8

# # Read Data File (Week 1)

#     Import Pandas

# In[1]:


import pandas as pd


#     Import Titanic Data

# In[2]:


df = pd.read_csv('titanic.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


#     Import Visualization Tools

# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import numpy as np


# In[8]:


import pandas_profiling


# # Numeric summaries:

# In[9]:


profile = df.profile_report(title='Titanic Profiling Report')


# In[10]:


profile


# In[11]:


profile.to_file("Titanic_profile.html")


# # Visual Summaries (Week 2: Replicate Profiling)

# In[12]:


df['pclass'].value_counts().plot(kind='barh')
for index, value in enumerate(df['pclass'].value_counts()):
    plt.text(value, index, str(value))
plt.title('Passenger Class Counts')
plt.show()


#     3rd Class makes up majority of passengers with 54.2%, then 1st Class with 24.7%, and then 2nd Class 21.2%

# Question: What class does crew belong to if any? 

# In[13]:


df['survived'].value_counts().plot(kind='barh')
for index, value in enumerate(df['survived'].value_counts()):
    plt.text(value, index, str(value))
plt.title('Survival Counts')
plt.show()


# In[14]:


df['sex'].value_counts().plot(kind='barh')
for index, value in enumerate(df['sex'].value_counts()):
    plt.text(value, index, str(value))
plt.title('Sex Counts')
plt.show()


# In[15]:


age = df['age']
plt.hist(age, density=False, bins=10)
plt.ylabel('Count')
plt.xlabel('age')
plt.title('Age Histogram')


# In[16]:


df['sibsp'].value_counts().plot(kind='barh')
for index, value in enumerate(df['sibsp'].value_counts()):
    plt.text(value, index, str(value))
plt.title('sibling Counts')
plt.show()


# In[17]:


df['parch'].value_counts().plot(kind='barh')
for index, value in enumerate(df['parch'].value_counts()):
    plt.text(value, index, str(value))
plt.title('Parents Plus Children Counts')
plt.show()


# In[18]:


age = df['fare']
plt.hist(age, density=False, bins=10)
plt.ylabel('Count')
plt.xlabel('Fare Paid')
plt.title('Fares Paid')


# In[19]:


df['embarked'].value_counts().plot(kind='barh')
for index, value in enumerate(df['embarked'].value_counts()):
    plt.text(value, index, str(value))
plt.title('Locations Embarked From')
plt.show()


# In[20]:


null_counter = df.isnull().sum()
null_df = pd.DataFrame({'null_counter':null_counter, 'name':null_counter.index})
null_df.plot(kind='barh',x = 'name', y = 'null_counter')


# In[21]:


df = df.fillna(0)


# In[22]:


df.plot(kind='scatter',x='age',y='age',color='blue')
plt.show()


# # Variable relationships:

# In[23]:


import seaborn as sns


# In[24]:


pearsoncorr = df.corr(method='pearson')
pearsoncorr


# In[25]:


sns.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu',
            annot=True,
            linewidth=0.5)


# In[26]:


spearmancorr = df.corr(method='spearman')
spearmancorr


# In[27]:


sns.heatmap(spearmancorr, 
            xticklabels=spearmancorr.columns,
            yticklabels=spearmancorr.columns,
            cmap='RdBu',
            annot=True,
            linewidth=0.5)


# In[28]:


kendallcorr = df.corr(method='kendall')
kendallcorr


# In[29]:


sns.heatmap(kendallcorr, 
            xticklabels=kendallcorr.columns,
            yticklabels=kendallcorr.columns,
            cmap='RdBu',
            annot=True,
            linewidth=0.5)


# In[30]:


#import phik
#from phik import resources, report
#df.corr()
#df.phik_matrix()


# In[31]:


sns.set(style="whitegrid")


# In[32]:


sexSurvived = df.pivot_table(index="sex", values="survived")
sexSurvived.plot.bar()
plt.show()


# In[33]:


classSurvived = df.pivot_table(index="pclass", values="survived")
classSurvived.plot.bar()
plt.show()


# # Models: 

#     Drop columns which are strings and not strongly correlated to survival. 

#     Model will cause errors if data not clean. 

# In[34]:


df.drop(['name', 'home.dest', 'boat', 'body', 'cabin', 'ticket'], axis = 'columns', inplace = True)


# Question: Even if variable is not strongly correlated, how can string variable be transformed into integer? 

#     Transforming string variables into integer variables. 

# In[35]:


df.sex = df.sex.map({'male': 0, 'female': 1})


# In[36]:


df.embarked = df.embarked.map({'S': 0, 'C': 1, 'Q': 2 })


#     Drop missing variables in entire dataset to prevent model error. 

# In[37]:


df = df.dropna()


# In[38]:


#Alternative: Replace missing variables with most popular type of variable or average varibale type.


# In[39]:


#df.embarked = df.embarked.fillna(0) 


# In[40]:


#df.age = df.age.fillna(inputs.age.mean()) 


# # Decision Tree Model (Week 3)

#     Define variables for tree model based on whole dataset.  

#     Inputs variable refers to "x-axis" of model: independent variable.

#     Target variable refers to "y-axis" of model: dependent variable. 

# In[41]:


inputs = df.drop(['survived'], axis = 'columns')
target = df.survived


#     Seperate dataset (titanic.csv) into test and train datasets: Train dataset ~ 75% and Test dataset ~ 25% of dataset. 

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size = 0.25)


#     Verify variables to model after cleaning of data complete. 

# In[44]:


inputs.head(5)


#     Create Decision Tree Model. 

# In[45]:


from sklearn import tree


# In[46]:


model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)


#     Verify accuracy of model. 

# In[47]:


model.score(X_test, Y_test)


#     Display Decision Tree as visual on Jupyter Notebook. 

# In[48]:


tree.plot_tree(model)


#     Print visual as .png file with pip install graphviz. File will open in new tab. 

# In[49]:


from graphviz import Source
graph = Source( tree.export_graphviz(model, out_file=None, feature_names=inputs.columns))
graph.format = 'png'
graph.render('model_render',view=True)

#     Feature Importance

importance = model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d Score: %.5f' % (i,v), )
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#	Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn.metrics
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(trainx, trainy)
forestpredict = clf.predict(testx)
print(confusion_matrix(testy, forestpredict))
print(classification_report(testy, forestpredict))

# # Logistic Regression Model (Week 4)

# In[ ]:




