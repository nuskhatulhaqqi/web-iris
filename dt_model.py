import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# link data 
dataset_url = 'https://raw.githubusercontent.com/nuskhatulhaqqi/data_mining/main/iris_data.csv'

# membaca data link
df = pd.read_csv(dataset_url)

X=df.iloc[:,0:4].values
y=df.iloc[:,4].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

hasil_prediksi = model.predict(X_test)

accuracy_dt=round(accuracy_score(y_test,hasil_prediksi)* 100, 2)

import pickle

with open('dt_pickle','wb') as r:
    pickle.dump(model,r)