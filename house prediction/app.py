import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, template_rendered
from xgboost import XGBRegressor
from sklearn import linear_model



app=Flask(__name__)

@app.route('/')
def web1():
   return render_template("index.html")
@app.route('/',methods=['POST'])
def web2():
   m=request.form['name']
   m=int(m)
   c=request.form['b']
   c=int(c)
   k={'AREA':[m],'BHK':[c]}
   df2=pd.DataFrame(k)
   l=request.form['se']
   f=l+".csv"
   
   data=pd.read_csv(f)
   #print(data)
   #print(data.values)
   New={'AREA':data['Area'],'BHK':data['bhk'],'PRICE':data['Price']}
   df1=pd.DataFrame(New)

   #print(df1)
   Y=df1['PRICE']
   X=df1.drop(['PRICE'],axis=1)

   X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=2)
   #print(X.shape,X_train.shape,X_test.shape)
   model=XGBRegressor()
   model.fit(X_train,Y_train)
   
   model2=linear_model.LinearRegression()
   model2.fit(X_train,Y_train)
   
   
   ka=model.predict(X_train)
   t=model.predict(df2)
   plt.scatter(Y_train,ka)
   plt.xlabel("actual_value")
   plt.ylabel("predicted_value")
   plt.title("graph for testing data")
   t=int(t)
   e=model2.predict(df2)
   e=int(e)
   minim=min(t,e)
   maxim=max(t,e)

   
   
   
   
      
   
   return render_template("web2.html",name=minim,name2=maxim,city=l,d=c,z=m)


if __name__ == '__main__':

   app.run(debug=True,port=5001)

