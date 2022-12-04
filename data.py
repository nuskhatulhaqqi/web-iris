import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('iris.csv')
data.head()

X=data.iloc[:,0:4].values
y=data.iloc[:,4].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#KNN DENGAN TUNING TETANGGA TERDEKAT 9

knn_clf = KNeighborsClassifier(n_neighbors=9)
knn_clf.fit(X_train,y_train)

y_pred = knn_clf.predict(X_test)

round(accuracy_score(y_test,y_pred),3)

import pickle

with open('knnnnnn_pickle','wb') as r:
    pickle.dump(knn_clf,r)