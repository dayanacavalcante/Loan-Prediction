# Loan Prediction

### _The Problem_
Dream Housing Finance is a home loan company. The company wants to automate the loan process by filling out an online form for the customer.

In this form the client must inform the following data: Sex, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others.

It is a classification problem as it will be foreseen whether the loan will be accepted or not.

Follow the link for the data:
https://www.kaggle.com/burak3ergun/loan-data-set?select=loan_data_set.csv

### _Exploratory Data Analysis_

Having a visual summary of the information makes it easier to identify patterns and trends than to look at the lines of a spreadsheet. For that I used seaborn which is a Python data visualization library based on matplotlib.

### _Transforming categorical variables into numeric ones_
To work with machine learning we need to transform categorical variables into numeric ones. For this I used the pandas get_dummies method.

### _Algorithms Used_
For this classification problem I used the following algorithms: Logistic Regression, Decision Tree and Random Forest.

### _Performance Metrics_
Through the F1_Score metric, it can be seen that Logistic Regression performed better than the three applied algorithms.



