import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# link data 
dataset_url = 'https://raw.githubusercontent.com/nuskhatulhaqqi/data_mining/main/iris_data.csv'

# membaca data link
df = pd.read_csv(dataset_url)

X=df.iloc[:,0:4].values
y=df.iloc[:,4].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#KNN DENGAN TUNING TETANGGA TERDEKAT 9

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
YY_pred = gaussian.predict(X_test)
accuracy_nb=round(accuracy_score(y_test,YY_pred)* 100, 2)


#PICKLE UNTUK MEMBUAT MODEL

import pickle

with open('nb_pickle','wb') as r:
    pickle.dump(gaussian,r)
