import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.ensemble import RandomForestClassifier

## load in titanic data
data = pd.read_csv('data/train.csv')
eval_data = pd.read_csv('data/test.csv')

def clean_data(data=[]):
    ## clean up data by dropping unremarkable columns,
    ## filling null data, and converting categorical data
    ## into numerical values
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'])
    return data

data = clean_data(data)

## split into training and test groups
split_idx = math.ceil(len(data) * .8)
train_data = data[:split_idx]
test_data = data[split_idx:]

## Split into features & label sets
X_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']

X_test = test_data.drop(['Survived'], axis=1)
y_test = test_data['Survived']

## train a random forest classifier
random_forest = RandomForestClassifier(n_estimators=50, max_features=10, max_depth=5)
random_forest.fit(X_train, y_train)
score = random_forest.score(X_test, y_test)
print('Score: %s'%score)

## create predictions off of evaluation data
X_eval = clean_data(eval_data)
Y_eval = random_forest.predict(X_eval)
results = pd.DataFrame.from_dict({
    "PassengerId": eval_data['PassengerId'],
    "Survived": Y_eval
})
results.to_csv('data/results.csv', index=False)

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, y_train)
# Y_pred = random_forest.predict(X_test)
# score = random_forest.score(X_train, y_train)
#
# results = pd.DataFrame.from_dict({
#     "PassengerId": test_data['PassengerId'],
#     "Survived": Y_pred
# })
#
# results.to_csv('data/results.csv', index=False)
