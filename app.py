from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np                        # numpy==1.19.3
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('agg')
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

app = Flask(__name__)

hasil = []

@app.route('/')
@app.route('/home')
def home():
    hasil.clear()
    return render_template('home1.html',title='Home')

@app.route('/profile')
def profile():
    hasil.clear()
    return render_template('profile.html',title='Profile')

@app.route("/dataset")
def dataset():
    df = pd.read_csv("iris_label.csv")
    return render_template("dataset.html", df_view = df)

def naive_bayes():
    #memanggil data
    data = pd.read_csv('iris.csv')
    #membagi data class dengan fitur
    X=data.iloc[:,0:4].values
    y=data.iloc[:,4].values
    # membagi data training dan testing
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    #naive bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    YY_pred = gaussian.predict(X_test)
    accuracy_nb=round(accuracy_score(y_test,YY_pred)* 100, 2)
    #buat model
    with open('nb_pickle','wb') as r:
        pickle.dump(gaussian,r)
    return accuracy_nb

def k_n_n():
    #memanggil data
    data = pd.read_csv('iris.csv')
    #membagi data class dengan fitur
    X=data.iloc[:,0:4].values
    y=data.iloc[:,4].values
    # membagi data training dan testing
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    #knn
    knn_clf = KNeighborsClassifier(n_neighbors=9)
    knn_clf.fit(X_train,y_train)
    y_pred = knn_clf.predict(X_test)
    accuracy=round(accuracy_score(y_test,y_pred)* 100, 2)
    with open('knnB_pickle','wb') as r:
        pickle.dump(knn_clf,r)
    return accuracy

def desicion_tree():
    #memanggil data
    data = pd.read_csv('iris.csv')
    #membagi data class dengan fitur
    X=data.iloc[:,0:4].values
    y=data.iloc[:,4].values
    # membagi data training dan testing
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    #desicion tree
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    hasil_prediksi = model.predict(X_test)
    accuracy_dt=round(accuracy_score(y_test,hasil_prediksi)* 100, 2)
    with open('dt_pickle','wb') as r:
        pickle.dump(model,r)
    return accuracy_dt

@app.route('/model')
def model():
    hasil.clear()
    a_nb = naive_bayes()
    a_knn = k_n_n()
    a_dt= desicion_tree()
    return render_template('model.html',title='Model',a_nb=a_nb,a_knn=a_knn,a_dt=a_dt)

def ubah_data(a):
    if a == 1:
        a='SETOSA'
    elif a == 2 :
        a = 'VERSICOLOR'
    else:
        a= 'VIRGINICA'
    return a

@app.route('/datas', methods=['POST','GET'])
def datas():
    if request.method == 'POST':
        if request.form['model'] == '1':
            akurasi= desicion_tree()
            jenis_model = "DESICION TREE"
            with open('dt_pickle', 'rb') as r:
                data = pickle.load(r)
        elif request.form['model'] == '2':
            akurasi= k_n_n()
            jenis_model = "KNN"
            with open('knnB_pickle', 'rb') as r:
                data = pickle.load(r)
        else:
            akurasi = naive_bayes()
            jenis_model = "NAIVE BAYES"
            with open('nb_pickle', 'rb') as r:
                data = pickle.load(r)

        sl = float(request.form['sl'])
        sw = float(request.form['sw'])
        pl = float(request.form['pl'])
        pw = float(request.form['pw'])
        datas = np.array((sl,sw,pl,pw))
        datas = np.reshape(datas, (1, -1))
        isBungaIris = data.predict(datas)
        with open('dt_pickle', 'rb') as r:
            model1 = pickle.load(r)
        with open('knnB_pickle', 'rb') as r:
            model2 = pickle.load(r)
        with open('nb_pickle', 'rb') as r:
            model3 = pickle.load(r)
        da = np.array((sl,sw,pl,pw))
        da = np.reshape(datas, (1, -1))
        dt = model1.predict(da)
        knn = model2.predict(da)
        nb = model3.predict(da)
        a=ubah_data(dt[0])
        b=ubah_data(knn[0])
        c=ubah_data(nb[0])
        hasil.append(a)
        hasil.append(b)
        hasil.append(c)
        finalData=isBungaIris
        return render_template('hasil.html',finalData=isBungaIris,a=hasil[0] ,b=hasil[1],c=hasil[2], data_model=jenis_model,akurasi=akurasi,hasil=hasil)
    else:
        hasil.clear()
        a_nb = naive_bayes()
        a_knn = k_n_n()
        a_dt= desicion_tree()
        return render_template('index.html',a_dt=a_dt,a_knn=a_knn,a_nb=a_nb,hasil=hasil)

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    Nama_Model= ['DESICION TREE','KNN','NAIVE BAYES']
    hasill=[hasil[0],hasil[1],hasil[2]]
    fig= plt.figure(figsize=(10,7))
    plt.bar(Nama_Model,hasill, color='lightcoral')
    plt.title('Hasil Perbandingan 3 Model', size=16)
    plt.ylabel('Jenis Bunga', size=14)
    plt.xlabel('Nama Model', size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    return fig

@app.route('/mod.png')
def mod_png():
    fig = create_figure2()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure2():
    Nama_Model= ['DESICION TREE','KNN','NAIVE BAYES']
    hasill=[desicion_tree(),k_n_n(),naive_bayes()]
    fig= plt.figure(figsize=(10,7))
    plt.bar(Nama_Model,hasill, color='lightcoral')
    plt.title('Grafik Perbandingan 3 Model', size=16)
    plt.ylabel('Akurasi', size=14)
    plt.xlabel('Nama Model', size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    return fig



if __name__ == "__main__":
    app.run(debug=True)
