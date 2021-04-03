import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def split_categorical_and_numerical_columns():
    categorical_columns = []
    numerical_columns = []
    for col in df.columns:
        if df[col].dtype.name == 'object':
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    return categorical_columns, numerical_columns


def replace_categorical_value(columns):
    for column in columns:
        unique_set = set(df[column].unique())   # Get unique set in Series
        try:
            unique_set.remove(-1)   # Remove -1 (nan) from set (if Series without -1 (nan) - pass)
        except KeyError:
            pass
        try:
            unique_set = sorted(unique_set)   # Sorting set to avoid random
        except TypeError:
            pass
        replaced_value = 0
        for value in unique_set:
            df.loc[df[column] == value, column] = replaced_value
            replaced_value += 1


df = pd.read_csv('weatherAUS.csv', delimiter=',')   # Read csv table
df.drop(columns=['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm'], inplace=True)  # Remove columns with priority nan
df['Date'] = pd.to_datetime(df['Date']).sub(pd.Timestamp(df.Date[0])).dt.days   # Categorical 'Date: 11-11-2009' to num

df.loc[pd.isnull(df['RainTomorrow']), 'RainTomorrow'] = 'No'   # This is should be necessary
categorical_columns, numerical_columns = split_categorical_and_numerical_columns()  # Split cat and num columns

for column in categorical_columns:
    df.loc[pd.isnull(df[column]), column] = -1  # Replace nan in cat_columns to -1 (to ez detect)

for column in numerical_columns:
    df.loc[pd.isnull(df[column]), column] = round(df[column].mean())   # Replace nan in numerical columns to mean value

replace_categorical_value(categorical_columns)

for column in categorical_columns:
    top_set_2 = set(df[column].value_counts()[:2].index.tolist())
    try:
        top_set_2.remove(-1)
    except KeyError:
        pass
    df.loc[df[column] == -1, column] = top_set_2.pop()  # Replace -1 (nan) in categorical columns to most common value


df = df.infer_objects()
x = df.iloc[:, :18]
y = df['RainTomorrow']

pd.to_numeric(y)
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.25, random_state=23)


classifier = BernoulliNB()
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print('BernoulliNB', classifier.score(x_test, y_test))
print(classification_report(prediction, y_test))

classifier = GaussianNB()
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print('GaussianNB', classifier.score(x_test, y_test))
print(classification_report(prediction, y_test))

classifier = LogisticRegression(penalty='l2', max_iter=100)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print('LogisticRegression', classifier.score(x_test, y_test))
print(classification_report(prediction, y_test))

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train, y_train)
prediction = neigh.predict(x_test)
print('KNeighborsClassifier', np.mean(prediction == y_test))
print(classification_report(prediction, y_test))
