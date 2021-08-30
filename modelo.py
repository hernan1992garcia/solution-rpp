import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import pickle

from scipy.stats import chi2_contingency
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from kmodes.kmodes import KModes
from matplotlib import pyplot as plt


#data must come in the format of dependent
def preparing_data(data):
    #Convert dcto in fraction
    data.loc[:,'dcto'] = data[['dcto','monto']].apply(lambda row:row[0]/row[1] if row[1]!=0 else 0,axis=1)
    #Assigning the new column "num_compras"
    data['num_compras']=data['ID_USER'].map(data['ID_USER'].value_counts())

    #Extracting information from column "dispositivo"
    #*Each dictionary key is considered as a realization of a categorical variable
    nam = list(eval(data.dispositivo.loc[0]).keys())
    list_data = (list(data.dispositivo.
                            apply(lambda x:
                            list(map(lambda x:str(x),list(eval(x).values())))
                            )))
    data.loc[:,nam] = pd.DataFrame(list_data,columns = nam)

    # Re escaling the data
    cols_num = ['monto','hora','linea_tc','interes_tc','dcto','cashback','num_compras']
    scaler = preprocessing.MinMaxScaler()
    data[cols_num] = scaler.fit_transform(data[cols_num])

    #The categorical columns 
    cols_categ = ['genero','establecimiento','ciudad','tipo_tc','status_txn','is_prime','model','device_score'
                ,'os']
    data_categ = data.loc[:,cols_categ].copy()

    #The feature model is an avoidable column
    data_categ = data_categ.drop('model',1)
    #establecimiento, ciudad and genero can be eliminated
    data_categ = data_categ.drop('establecimiento',1)
    data_categ = data_categ.drop('ciudad',1)
    data_categ = data_categ.drop('genero',1)

    #The numerical columns
    cols_num = ['monto','hora','linea_tc','interes_tc','dcto','num_compras']
    data_num = data.loc[:,cols_num]

    # Numerical and categorical data with the modifications made before are unified.
    data_f = data_categ.copy()
    cols_num = list(data_num.columns)
    data_f[cols_num] = data_num

    #Dummy variables
    categ_cols = ['tipo_tc','status_txn','is_prime','device_score','os']
    X = pd.get_dummies(data_f,columns = categ_cols)
    ids = data.ID_USER
    return {'data':X,'ids':ids}



def eval_model(X,ids):
    clf2 = pickle.load(open('model.pkl', 'rb'))
    y_pred = clf2.predict(X)
    pred_df = pd.DataFrame(columns = ['ID_USER','fraude'])
    pred_df.loc[:,'ID_USER'] = ids
    pred_df.loc[:,'fraude'] = y_pred
    pred_df.loc[:,'fraude'] = pred_df.fraude.apply(lambda x:x==1)

    return pred_df
