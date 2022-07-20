from flask import Flask,render_template,url_for,request,jsonify
import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split as tt
from sklearn.preprocessing import MinMaxScaler as mn 
from sklearn.metrics import r2_score

import joblib
import pickle

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/batting')
def batting():
    df=pd.read_csv("bat.csv")
    x=df[['ipl_Inns','ipl_Runs','ipl_Avg','ipl_SR','ipl_6s','ipl_100','ipl_50','ipl_BF']]
    y=df[['Amount']]
    m=mn()
    df_sc=m.fit_transform(x)
    df_sc_df=pd.DataFrame(df_sc,columns=x.columns,index=df.index)
    x=df_sc_df
    x_train,x_test,y_train,y_test=tt(x,y,test_size=0.2,random_state=42)
    xgb_r = xg.XGBRegressor(n_estimators = 503,eta=0.12,max_depth=7,subsample=0.1,colsample_bytree=1,colsample_bylevel=1,alpha=0.7,num_parallel_tree=8)
    xgb_r.fit(x_train, y_train)  
    pickle.dump(xgb_r,open("model.pkl","wb"))
    ypred=xgb_r.predict(x_test)
    #r=r2_score(y_test,ypred)
    return render_template('batting.html')

@app.route('/batstats')
def batstats():
    return render_template('batstats.html')

@app.route('/bowlstats1')
def bowlstats1():
    return render_template('bowlstats1.html')
#ML routes


@app.route('/predict', methods=['GET','POST'])
def predict():
    v1 = int(request.form['var1'])
    v2 = int(request.form['var2'])
    v3 = float(request.form['var3'])
    v4 = float(request.form['var4'])
    v5 = int(request.form['var5'])
    v6 = int(request.form['var6'])
    v7 = int(request.form['var7'])
    v8 = int(request.form['var8'])
    form_array=np.array([[v1,v2,v3,v4,v5,v6,v7,v8]])
    mdl=pickle.load(open("model.pkl","rb"))
    #m=mn()
    #df_sc=m.fit_transform(form_array)
   

    pred=mdl.predict(form_array)[0]

        #pred = xgb_r.predict([[v1,v2,v3,v4,v5,v6,v7,v8]])

    return render_template('batting.html',prediction=pred)
 
    




    '''if request.method == 'POST':
        v1 = int(request.form['var1'])
        v2 = int(request.form['var2'])
        v3 = float(request.form['var3'])
        v4 = float(request.form['var4'])
        v5 = int(request.form['var5'])
        v6 = int(request.form['var6'])
        v7 = int(request.form['var7'])
        v8 = int(request.form['var8'])
        form_array=np.array([[v1,v2,v3,v4,v5,v6,v7,v8]])
        mdl=pickle.load(open("model.pkl","rd"))
        pred=mdl.predict(form_array)[0]

        #pred = xgb_r.predict([[v1,v2,v3,v4,v5,v6,v7,v8]])

    return render_template('index.html',prediction=pred)'''


if __name__ == '__main__':
    app.run(debug=True)