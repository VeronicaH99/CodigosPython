import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
import pyodbc

def normalizar(X):
    m,n = np.shape(X)
    minimos = np.min(X,axis=0)
    maximos = np.max(X,axis=0)
    Xn=(X-minimos)/(maximos-minimos)
    return Xn

def validacion(X,Y):
    m,n = X.shape
    corte70 = round(m*0.7)
    sorteo = np.random.permutation(m)
    Xtrain = X[sorteo[0:corte70],:]
    Xtest = X[sorteo[corte70:],:]
    Ytrain = Y[sorteo[0:corte70],:]
    Ytest = Y[sorteo[corte70:],:]
    return Xtrain,Ytrain,Xtest,Ytest

def readsql(driver,server,database,username,password,SQL_datos):
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    return cnxn