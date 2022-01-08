import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('trainingData.csv',usecols=[str(i) for i in range(21)]+['Label'])

for i in range(2000):
    if df['Label'][i]=='stop':
        df['Label'][i]=1
    elif df['Label'][i]=='up':
        df['Label'][i]=2
    elif df['Label'][i]=='victory':
        df['Label'][i]=3
    elif df['Label'][i]=='yes':
        df['Label'][i]=4

y = df['Label']
y = y.astype('int')
X = df.drop('Label',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))

rfc.fit(X,y)

pickle.dump(rfc, open('firstModel.sav', 'wb'))