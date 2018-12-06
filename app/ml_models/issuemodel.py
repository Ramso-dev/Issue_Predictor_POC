#file  -- issuemodel.py --
import os 
import json
import numpy as np
import pandas as pd
import time
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#print(os.listdir("../flask_api/data"))

#ENV
datapath = '../issue_predictor_poc/app/ml_datasets/'

nice_data = pd.read_csv(datapath + 'nice_data.csv',index_col='issueid') #we use this to get an original form of the issues. We will filter it using the test set's indexes 
train = pd.read_csv(datapath + 'train_refined.csv',index_col='issueid')
test = pd.read_csv(datapath + 'test_refined.csv',index_col='issueid')


#This columns showed useless in the feature extraction phase
to_drop = ['openingtime','component','os','updates','cluster','0','1','2','3']
train.drop(to_drop, axis=1, inplace=True)
test.drop(to_drop, axis=1, inplace=True)

#n-rows has to be the same
#print('Are rows consistent between datasets?', (X_data.shape[0] == y_data.shape[0]) and (X_data.shape[0] == nice_data.shape[0])) 


#Global vars for model
model = None
gX_train = train.iloc[:,1:]
gX_test = test.iloc[:,1:]
gy_train = train['duration']
gy_test = test['duration']
gidx_train = list(train.index.values)
gidx_test = list(test.index.values)
gbase_mae = 0




from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from xgboost import XGBRegressor


    
def XGB_train(X_train,y_train,X_test,y_test):
    
    #indexes = X_data.index.values
    
    print('Preprocessing data')
    
    print('removing low participation devs')
    X_train ,y_train = removeDevsLow(X_train,y_train)
    print(X_train.shape)
    
    print('removing outliers')
    X_train ,y_train = removeOutliers(X_train,y_train)
    
    print('get dummies')
    X_train = getDummies(X_train)
    X_test = getDummies(X_test)
          
    print('get missing columns')      
    # Get missing columns in the training test
    missing_cols = list(set(X_train.columns) - set(X_test.columns))
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        X_test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    X_test = X_test[X_train.columns]
    
    test=pd.concat([y_train, X_train], axis=1)
    corrMatt = test.corr()
    lowcorrelation=corrMatt[(corrMatt['duration']<0.06)&(corrMatt['duration']>-0.06)] #Best value: 0.1
    #print(lowcorrelation)

    labs = lowcorrelation.index.values
    X_train.drop(labs, axis=1, inplace=True)
    X_test.drop(labs, axis=1, inplace=True)

    print('Dropped', len(labs), 'low correlated columns')

    global gX_train
    gX_train = X_train
    global gX_test
    gX_test = X_test
    
    
    #Baseline errors MAE Score
    baseline_preds = y_test.mean()
    baseline_errors = abs(baseline_preds - y_test)
    base_mae = round(np.mean(baseline_errors), 2)
    print('Average baseline error: ', base_mae)
    
    print(baseline_errors.describe())
    
    
    global gbase_mae
    gbase_mae = base_mae
    
    #Standarize
    y_train = np.log(y_train)
    
    print('Training XGB')

    feature_selector = VarianceThreshold()
    X_train = feature_selector.fit_transform(X_train)
    X_test = feature_selector.transform(X_test)


    regressor = XGBRegressor()
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                  'objective':['reg:linear'],
                  'learning_rate': [.03], #so called `eta` value
                  'max_depth': [8],
                  #'max_depth': [7,8,9],
                  #'min_child_weight': [9],
                  'silent': [1],
                  'subsample': [0.7],
                  'colsample_bytree': [0.5],
                  #'colsample_bytree': [0.5,0.7,1],
                  'n_estimators': [200]} #200
    xgb_grid = GridSearchCV(regressor,
                            parameters,
                            cv = 2,
                            #n_jobs = 5,
                            verbose=True)


    
    xgb_fit = xgb_grid.fit(X_train, y_train)
    
    
    #print(xgb_grid.best_params_)
    #print(xgb_grid.best_score_)

    print('Training completed')
    
    return xgb_fit  #,X_test,y_test # , idx_train, idx_test
    
def removeDevsLow(data,y):
    data = pd.concat([data, y], axis=1)
    to_remove = data['devs'].value_counts()[data['devs'].value_counts() < 150].index
    data['devs'].replace(to_remove, np.nan, inplace=True)
    data['devs'].replace(np.NaN, 'low', inplace=True)
    data = data[data['devs'] != 'low']
    
    return data.iloc[:,0:-1],data.iloc[:,-1]

def getDummies(data): 
    if 'severity' in data:
        dummies_sev = pd.get_dummies(data['severity'])
    if 'priority' in data:
        dummies_pri = pd.get_dummies(data['priority'])
    if 'devs' in data:
        dummies_dev = pd.get_dummies(data['devs'])
    if 'component' in data:
        dummies_comp = pd.get_dummies(data['component'],drop_first =True)

    data_dum = pd.DataFrame()
    if 'severity' in data:
        data_dum = pd.concat([data_dum, dummies_sev], axis=1)
    if 'priority' in data:
        data_dum = pd.concat([data_dum, dummies_pri], axis=1)
    if 'devs' in data:
        data_dum = pd.concat([data_dum, dummies_dev], axis=1)
    if 'component' in data:
        data_dum = pd.concat([data_dum, dummies_comp], axis=1)
    
    #issue types
    data_dum = pd.concat([data_dum, data.iloc[:,4:]], axis=1)
    return data_dum

def removeOutliers(data,y):
    data = pd.concat([data, y], axis=1)
    data = data[data.loc[:,'duration']<0.5*(10**8)]
    return data.iloc[:,0:-1],data.iloc[:,-1]

    
def setGlobalModel():
    #if __name__ == '__main__':
    xgb = XGB_train(gX_train,gy_train,gX_test,gy_test)
    global model
    model = xgb
    
    
def predict(X,idx): #predicts using an array of values and its indexes
    
    predictions = model.predict(X.values)

    predictions = np.exp(predictions)
    print('AAA',max(set(list(predictions)), key=list(predictions).count))
    #predictions.apply(enhance)
    
    
    #enhance = lambda t: t * 7 if t > 984748.25 else t*1
    #vfunc = np.vectorize(enhance)
    #predictions = vfunc(predictions)
    
    #predictions = np.expm1(predictions) #from normalization
    
    #errors = abs(predictions - y_test.values)
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.', 'max:',max(errors.values),'min',min(errors.values))
    
    
    predictions_df = pd.DataFrame(data=predictions, index=idx, columns=['prediction'])
    
    
    
    #print(X.values,idx)
    #print(predictions_df)
    #print(gy_test)
    #score_models_mae(predictions_df, gy_test)
    errors = abs(predictions - gy_test[idx])
    pred_mae = round(np.mean(errors), 2)
    print('pred_mae score: ', pred_mae)
    
    improvement=(gbase_mae-pred_mae)*100/gbase_mae
    print('Improvement relative to base_mae:',improvement,'%')
    
    print(errors.describe())
    
    #print(predictions_df.shape)
    #print(predictions_df)
    
    #we get the original issues filtering nice_data with the test set's indexes
    issues_df = nice_data.ix[idx]
    print(issues_df.shape)
    #print(issues_df)
    
    issues_df['prediction']=predictions_df['prediction'].div(86400).round(2) #transform into days
    issues_df['issueid']=issues_df.index
    issues_df['duration']=issues_df['duration'].div(86400).round(2) #transform into days
    issues_df['openingdate']=issues_df['openingtime'].apply(epochToDate)
    
    

    return issues_df.astype(object),round(gbase_mae/86400,2),round(pred_mae/86400,2) #needed to convert to dict later

def epochToDate(epochtime):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epochtime))

def score_models_mae(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        errors = abs(P.loc[:, m].values - y)
        score = round(np.mean(errors), 2)
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

setGlobalModel()

print('-------------model---------------')

#predict(gX_test.iloc[8:12,:],gidx_test[8:12])
predict(gX_test,gidx_test)


