#IMPOERTAMOS LIBRERIAS GENERALES
import pandas as pd
# import numpy as np
import pyodbc
import json
from FUNCIONES import normalizar

#IMPORTAR LIBRERIAS DEL ALGORITMO
from sklearn import svm
from sklearn.metrics import classification_report

#LIBRERIAS DE VALIDACION
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

#IMPORTAR LIBRERIAS PARA EVALUAR DESEMPEÑO
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import accuracy_score

#CARGAR MATRIZ DE CARACTERISTICAS Y VESTOR DE CLASES DESDE SQL
driver = "ODBC Driver 17 for SQL Server"
server = "servidornuevo1.database.windows.net"
database = "basededatos"
username = "servidornuevo"
password = "ITM_2020"

#SE ESTABLECE CONEXION
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor() 

#LEER Y
Yquery = "SELECT [Y] FROM logsY;"
YSQL = pd.read_sql(Yquery, cnxn)
#LEER X
Xquery = "SELECT [X1],[X2],[X3],[X4],[X5],[X6],[X7],[X8],[X9],[X10],[X11],[X12] FROM logsX;"
XSQL = pd.read_sql(Xquery, cnxn)

#CONVERTIR DATAFRAME DE PANDAS A MATRIZ NUMPY
Y=YSQL.values
X=XSQL.values

#NORMLIZAR
Xn=normalizar(X)

#VALIDACION 70/30
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xn,Y,test_size=0.3,random_state=42)

#VALIDACION K_FLOD
# kf = KFold(n_splits=2)
# for train_index, test_index in kf.split(Xn):
#     #print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = Xn[train_index], Xn[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
    
#ENTRENAR SVM
modelo = svm.SVC(kernel='linear',gamma='auto')
modelo.fit(Xtrain, Ytrain)
Yes = modelo.predict(Xtest)

# f1_score=np.mean(f1_score(Ytest,Yes,average=None))
# precision=np.mean(precision_score(Ytest, Yes, average=None))
# recall=np.mean(recall_score(Ytest, Yes, average=None))
# accuracy=accuracy_score(Ytest, Yes)

json_response = json.dumps(classification_report(Ytest, Yes),indent=2)
json_response = classification_report(Ytest, Yes,labels=[-1,0,1])