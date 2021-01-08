# Imports
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Loading Data
data = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\Loan Prediction\\Data\\loan_data_set.csv')
print(data.head())

print(data.shape)

# Data Preparation
print(data.isnull().sum())

data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)

print(data['LoanAmount'].median())
data['LoanAmount'].fillna(128,inplace=True)

print(data['Loan_Amount_Term'].value_counts())
data['Loan_Amount_Term'].fillna(360,inplace=True)

print(data.isnull().sum())

print(data.apply(lambda x: len(x.unique())))

# Exploratory Data Analysis
#%%
sns.countplot(x='Gender', hue='Loan_Status', data=data, palette='Set3')
#%%
sns.countplot(x='Married', hue='Loan_Status', data=data, palette='Set3')
#%%
sns.countplot(x='Self_Employed', hue='Loan_Status', data=data, palette='Set3')
#%%
sns.countplot(x='Credit_History', hue='Loan_Status', data=data, palette='Set3')
#%%
sns.countplot(x='Property_Area', hue='Loan_Status', data=data, palette='Set3')
#%%
sns.countplot(x='Loan_Amount_Term', hue='Loan_Status', data=data, palette='Set3')

# Separate the variables between predictors and target
#%%
x = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']
#%%
print(x.head())
print(y.head())

# Transforming categorical variables into numeric ones
#%%
x = pd.get_dummies(x)
print(x.head())

y = data['Loan_Status'].replace('Y', 1).replace('N', 0)
print(y.head())

# Create the training and test dataset
#%%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3,random_state=2)

print(x_train.shape)
print(y_train.shape)

# Model Training
## Logistic Regression
#%%
model=LogisticRegression()
model.fit(x_train,y_train)

## Decision Tree
#%%
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)

## Random Forest Classifier
#%%
forest=RandomForestClassifier()
forest.fit(x_train,y_train)

# Performance Metrics
## Logistic Regression
#%%
ypred=model.predict(x_test)
print(ypred)

evaluation=f1_score(y_test,ypred)
print(evaluation)

### Confusion Matrix
#%%
true_labels=y_test
pred_labels=ypred

confusion_matrix=confusion_matrix(true_labels,pred_labels)
sns.heatmap(confusion_matrix,annot=True,fmt='g',cmap=plt.cm.Greens)
plt.title("Confusion Matrix Logistic Regression")
ticks=np.arange(2)
plt.xticks=(ticks,ticks)
plt.yticks=(ticks,ticks)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()

## Decision Tree
#%%
ypred_tree=tree.predict(x_test)
print(ypred_tree)

evaluation_tree=f1_score(y_test,ypred_tree)
print(evaluation_tree)

### Confusion Matrix
#%%
true_labels=y_test
pred_labels=ypred_tree

confusion_matrix=confusion_matrix(true_labels,pred_labels)
sns.heatmap(confusion_matrix,annot=True,fmt='g',cmap=plt.cm.Greens)
plt.title("Confusion Matrix Decision Tree")
ticks=np.arange(2)
plt.xticks=(ticks,ticks)
plt.yticks=(ticks,ticks)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()

## Random Forest Classifier
#%%
ypred_forest=forest.predict(x_test)
print(ypred_forest)

evaluation_forest=f1_score(y_test,ypred_forest)
print(evaluation_forest)

### Confusion Matrix
#%%
true_labels=y_test
pred_labels=ypred_forest

confusion_matrix=confusion_matrix(true_labels,pred_labels)
sns.heatmap(confusion_matrix,annot=True,fmt='g',cmap=plt.cm.Greens)
plt.title("Confusion Matrix Random Forest Classifier")
ticks=np.arange(2)
plt.xticks=(ticks,ticks)
plt.yticks=(ticks,ticks)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()
# %%