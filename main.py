import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pdb

## Load in Training Data and Testing Data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

## Drop uncessary columns
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_data = test_data.drop(['Name', 'Ticket'], axis=1)

## Clean up NaN Data
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')

# ## Display Histogram of Ages
# data.Age.hist()
# plt.show()
#
# ## Find a correlation in the data
# corr = data.corr()
# fig, ax = plt.subplots(figsize=(16, 12))
# fig = sns.heatmap(corr,annot=True)
# plt.show()
#
# ## plot embarked vs survived
# sns.factorplot('Embarked', 'Survived', data=train_data, size=4, aspect=3)
# plt.show()
#
# ## plot age vs survived
# sns.factorplot('Age', 'Survived', data=train_data, size=4, aspect=3)
# plt.show()
#
# ## plot pclass vs survived
# sns.factorplot('Pclass', 'Survived', data=train_data, size=4, aspect=3)
# plt.show()

train_data.drop(['Embarked', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Embarked', 'Cabin'], axis=1, inplace=True)

# def get_person(passenger):
#     age,sex = passenger
#     return 'child' if age < 16 else sex
#
# train_data['Person'] = train_data[['Age', 'Sex']].apply(get_person, axis=1)
# test_data['Person'] = test_data[['Age', 'Sex']].apply(get_person, axis=1)

train_data.drop(['Age', 'Sex', 'Parch', 'SibSp'], axis=1, inplace=True)
test_data.drop(['Age', 'Sex', 'Parch', 'SibSp'], axis=1, inplace=True)

train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

X_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']
X_test = test_data.drop("PassengerId", axis=1).copy()

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
score = random_forest.score(X_train, y_train)

results = pd.DataFrame.from_dict({
    "PassengerId": test_data['PassengerId'],
    "Survived": Y_pred
})

results.to_csv('data/results.csv', index=False)
