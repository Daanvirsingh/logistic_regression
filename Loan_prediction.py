from sklearn import linear_model 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn import metrics
dirname = os.path.dirname(__file__)
def multiEnc(list):
    for i in list:
        data[i]=data[i].astype("category")
        data[i]=data[i].cat.codes
    
def multifillna(list):
    for i in list:
        data[i]=data[i].astype("float")
        data[i] = data[i].fillna((data[i].mean()))

#Extracting the data
data=pd.read_csv(os.path.join(dirname, "train_u6lujuX_CVtuZ9i.txt"),sep=",",header=None)
header=data.iloc[0]
data=data[1:]
data=data.rename(columns=header)
yy=data['Loan_ID']
data=data.drop(['Loan_ID'],axis=1)

#Encoding all values
multiEnc(["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status","Dependents"])

#filling missing values
multifillna(data.columns)

#normalization
data=(data-data.min())/(data.max()-data.min())


y=data["Loan_Status"]
data=data.drop(["Loan_Status"],axis=1)
d=data
#Logistic Regression model learn
linear=linear_model.LogisticRegression()
linear.fit(data,y)
linear.score(data,y)

# getting test data
data=pd.read_csv(os.path.join(dirname, "test_Y3wMUE5_7gLdaTN.txt"),sep=",",header=None)
header=data.iloc[0]
data=data[1:]
data=data.rename(columns=header)
x=data["Loan_ID"].values
data=data.drop(['Loan_ID'],axis=1)
multiEnc(["Gender","Married","Education","Self_Employed","Property_Area","Dependents"])
multifillna(data.columns)
data=(data-data.min())/(data.max()-data.min())

#predicting test data
y_pred=linear.predict(data)
y_pred=y_pred.astype(int)
data=pd.DataFrame(np.column_stack((x,y_pred)),columns=['Loan_ID','Loan_Status'])
data['Loan_Status']=data['Loan_Status'].astype(str)
data['Loan_Status']=data['Loan_Status'].replace(str(1),"Y")
data['Loan_Status']=data['Loan_Status'].replace(str(0),"N")
#creating csv of result predictions
#print(metrics.accuracy_score(y_pred,y))
data.to_csv(os.path.join(dirname,'yolo.csv'),index=False)