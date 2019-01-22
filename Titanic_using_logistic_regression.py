
# The following code was written in Jupyter notebook. Please excuse uneven spacing. 
# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style

# In[2]:
from sklearn.linear_model import LogisticRegression

# In[3]:
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv") 

# In[4]:
passengerId = test_df["PassengerId"]

# In[5]:
train_df.head()

# In[6]:
# Dropping Passenger ID and Tricket
train_df = train_df.drop(['PassengerId'], axis = 1)
train_df = train_df.drop(['Ticket'], axis = 1)
test_df = test_df.drop(['PassengerId'], axis = 1)
test_df = test_df.drop(['Ticket'], axis = 1)

# In[7]:
train_df.info()

# In[8]:
# Cleaning the Deck data; Lower value implies lesser significance
data = [train_df, test_df]
def cabin_(x):
    if x == 'A':
        return 8
    elif x == 'B':
        return 7
    elif x == 'C':
        return 6
    elif x == 'D':
        return 5
    elif x == 'E':
        return 4
    elif x == 'F':
        return 3
    elif x == 'G':
        return 2
    elif x == 'T':
        return 1
    else:
        return 0
for k in data:
    k['Deck'] = k['Cabin'].str.slice(0,1).apply(cabin_)

# In[9]:
train_df = train_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis = 1)

# In[10]:
train_df.head(10)

# In[11]:
# Cleaning up the port of Embarkment
data = [train_df, test_df]
def embarkment_(x):
    if x == 'S':
        return 1
    elif x == 'C':
        return 2
    elif x == 'Q':
        return 3
    else:
        return 1
for k in data:
    k['Embarked'] = k['Embarked'].apply(embarkment_)

# In[12]:
train_df.head()

# In[13]:
# Getting the Title of the name; Lower values implies higher significance
data = [train_df, test_df]
def name_(x):
    x1 = x.split(',')[1].split('.')[0][1:]
    if (x1 == 'Mr') or (x1 == 'Don'):
        return 'Mr'
    elif (x1 == 'Mrs') or (x1 == 'Mme') or (x1 == 'Dona'):
        return 'Mrs'
    elif (x1 == 'Miss') or (x == 'Mlle') or (x == 'Ms'):
        return 'Miss'
    elif x1 == 'Master':
        return 'Master'
    elif (x1 == 'Capt') or (x1 == 'Capt') or (x1 == 'Col') or (x1 == 'Major'):
        return 'Officers'
    else: 
        return 'Royalty'
for k in data:
    k['Prefix'] = k['Name'].apply(name_)
train_df = train_df.drop(['Name'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)

# In[14]:
# Creating the mean of ages by Prefix
ages = {}
Prefixes = train_df['Prefix'].unique()
for k in Prefixes:
    ages[k] = train_df[(train_df.Prefix == k) & (~train_df.Age.isnull())]['Age'].mean()

# In[15]:
data = [train_df, test_df]
def ageFill_(x):
    return ages[x]
for k in data:
    k['AgesNew'] = k['Prefix'].apply(ageFill_)

# In[16]:
test_df['Age'] = test_df['Age'].fillna(test_df['AgesNew'])
train_df['Age'] = train_df['Age'].fillna(train_df['AgesNew'])

# In[17]:
train_df = train_df.drop(['AgesNew'], axis = 1)
test_df = test_df.drop(['AgesNew'], axis = 1)

# In[18]:
# Converting sex to a numeric value; male = 1, female = 2;
data = [train_df, test_df]
def sex_(x):
    if x == 'male':
        return 1
    else:
        return 2
for k in data:
    k['Sex'] = k['Sex'].apply(sex_)

# In[19]:
# Indicating whether the passenger was a child. Anyone who is under 18 years is considered a child. 1 for a child, 0 for a adult
data = [train_df, test_df]
def child_(x):
    if x < 18:
        return 1
    else:
        return 0
for k in data:
    k['Child'] = k['Age'].apply(child_)

# In[20]:
# Indicating whether the passenger had a family. 1 indicates no family, anything greater is the number of family members
data = [train_df, test_df]
def family_(x):
    if x > 0:
        return x
    else:
        return 0
for k in data:
    k['Family'] = k['SibSp'].apply(family_) + k['Parch'].apply(family_) +1

# In[21]:
data = [train_df, test_df]
def alone_(x):
    if x > 1:
        return 0
    else:
        return 1
for k in data:
    k['Alone'] = k['Family'].apply(alone_)

# In[22]:
train_df.head()

# In[23]:
train_df.describe()

# In[24]:
data = [train_df, test_df]
for k in data:
    k['Fare'] = k['Fare']/k['Family']

# In[25]:
data = [train_df, test_df]
def name1_(x1):
    if (x1 == 'Mr') or (x1 == 'Don'):
        return 1
    elif (x1 == 'Mrs') or (x1 == 'Mme') or (x1 == 'Dona'):
        return 3
    elif (x1 == 'Miss') or (x1 == 'Mlle') or (x1 == 'Ms'):
        return 3
    elif x1 == 'Master':
        return 2
    elif (x1 == 'Capt') or (x1 == 'Capt') or (x1 == 'Col') or (x1 == 'Major'):
        return 1
    else: 
        return 2
for k in data:
    k['Prefix'] = k['Prefix'].apply(name1_)

# In[26]:
data = [train_df, test_df]
def Fare_(x):
    if x <= 7.91:
        return 0
    elif (x > 7.91) and (x <= 14.454):
        return 1
    elif (x > 14.454) and (x <= 31):
        return 2
    else: 
        return 3
def Age_(x):
    if x <= 16:
        return 0
    elif (x > 16) and (x <= 32):
        return 1
    elif (x > 32) and (x <= 48):
        return 2
    elif (x > 48) and (x <= 64):
        return 3
    else: 
        return 4
for k in data:
    k['Age'] = k['Age'].apply(Age_)
    k['Fare'] = k['Fare'].apply(Fare_)

# In[27]:
train_df.head()

# In[28]:
test_df['Fare'] = test_df['Fare'].fillna(0)

# In[29]:
# Building the model
# In[30]:
X_train = train_df.drop(["Survived"], axis=1)
Y_train = train_df["Survived"]
X_test  = test_df

# In[31]:
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Pred = lr.predict(X_test)

# In[32]:
lr.score(X_train, Y_train)

# In[33]:
d = {'PassengerId':passengerId, 'Survived':Pred}

# In[34]:
z = pd.DataFrame(data=d)

# In[35]:
z.to_csv('Result.csv', index = False)


# from sklearn.svm import SVC
# clf = SVC()
# clf.fit(X_train, Y_train)
# Pred = clf.predict(test_df)
# train_df.to_csv('c1train.csv', index = False)
# test_df.to_csv('c1test.csv', index = False)


# In[616]:


# from sklearn.ensemble import RandomForestClassifier


# In[622]:


# from sklearn.model_selection import cross_val_score
# rf = RandomForestClassifier(n_estimators=100)
# scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

# print("Scores:", scores)
# print("Mean:", scores.mean())
# print("Standard Deviation:", scores.std())
# rf.fit(X_train, Y_train)
# rf.score(X_train, Y_train)
# pred = rf.predict(X_test)
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
# accuracy=round(knn.score(X_train,Y_train)*100,2)
# accuracy
# from sklearn.ensemble import RandomForestClassifier
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_prediction = random_forest.predict(X_test)
# random_forest.score(X_train, Y_train)
# round(random_forest.score(X_train, Y_train) * 100, 2)
# decision_tree = DecisionTreeClassifier() 
# decision_tree.fit(X_train, Y_train)  
# Y_pred = decision_tree.predict(X_test)  
# round(decision_tree.score(X_train, Y_train) * 100, 2)
# perceptron = Perceptron(max_iter=120)
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

