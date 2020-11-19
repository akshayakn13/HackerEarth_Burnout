# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:37:44 2020

@author: aksha
"""

from joblib import dump, load
import pandas as pd 
def predict(test):
    ids = test["Employee ID"].values
    test.drop(["Employee ID","Date of Joining"],axis = "columns",inplace = True)
    model = load('D:\\firefox downloads\\Hackerearth ML\\dataset\\gbdt_model.joblib')
    #gender = load('D:\\firefox downloads\\Hackerearth ML\\dataset\\gender.joblib')
    #company = load("D:\\firefox downloads\\Hackerearth ML\\dataset\\company.joblib")
    #wfh = load("D:\\firefox downloads\\Hackerearth ML\\dataset\\wfh.joblib")
    std = load('D:\\firefox downloads\\Hackerearth ML\\dataset\\std.joblib')
    #gender_code = gender.transform(test["Gender"]).reshape(-1,1)
    #company_code = company.transform(test["Company Type"]).reshape(-1,1)
    #wfh_code = wfh.transform(test["WFH Setup Available"]).reshape(-1,1)
    test = pd.get_dummies(data = test,columns = ["Gender","Company Type","WFH Setup Available"])
    #mental = std.transform(test["Mental Fatigue Score"].values.reshape(-1,1))
    mental_test = std.transform(test["Mental Fatigue Score"].values.reshape(-1,1))
    test["std_mental"] = mental_test
    test.drop("Mental Fatigue Score",axis = "columns",inplace = True)
    prediction = model.predict(test)
    df = pd.DataFrame(list(zip(ids,prediction)),columns=["Employee ID","Burn Rate"])
    df.to_csv("D:\\firefox downloads\\Hackerearth ML\\dataset\\prediction_gbdt.csv",index = False)

if __name__ == '__main__':
    df = pd.read_csv("D:\\firefox downloads\\Hackerearth ML\\dataset\\test.csv")
    predict(df)