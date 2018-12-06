#file  -- main2.py --
from app.ml_models import issuemodel
#from model import predict
#from model import X_test,idx_test
#gX_test,gidx_test
from flask import Flask, request
from flask_jsonpify import jsonify
from flask_restful import Resource, Api
import json
from json import dumps


import datetime

import pandas as pd


print("Starting predictor...")


X_data = pd.read_csv('../issue_predictor_poc/app/ml_datasets/X_data.csv')
#X_data.set_index('issueid', inplace=True)
y_data = pd.read_csv('../issue_predictor_poc/app/ml_datasets/y_data.csv')
X_data = X_data.iloc[0:20,:]
y_data = y_data.iloc[0:20,:]
#print(data)



class Issues(Resource):
    def get(self):
        
        #result=X_data.to_dict('records')
        #print(X_data)
        result=X_test.to_dict('index')
        print(X_test)
        return result
        #return jsonify(result)

class Predictions(Resource):
    def get(self):
        result,base_mae,pred_mae = issuemodel.predict(issuemodel.gX_test,issuemodel.gidx_test)
        #result,base_mae,pred_mae = modelv2.predict(modelv2.gX_test.iloc[40:80,:],modelv2.gidx_test[40:80])
        
        result['errors']= abs(result['prediction'] - result['duration'])
        
        
        #result['name']=result['openingdate']
        result['name']=result['openingtime'].apply(toDate)
     
        result['value']=result['prediction']
        #print(result.head())
        result_dict=result.to_dict('records')
        keys = ['name', 'value']
        newkeys = ['name','value']
        #result = {x:result[x] for x in keys
        #print(len(result))

        result2=[]
        for x in range(len(result_dict)):
            extra={k:result_dict[x][k] for k in keys }
            #for n in newkeys:
            #    extra[n] = extra.pop(k)
            result2.append(extra)
            
        result['value']=result['duration']
        result3=[]
        result_dict=result.to_dict('records')
        for x in range(len(result_dict)):
            extra={k:result_dict[x][k] for k in keys }
            #for n in newkeys:
            #    extra[n] = extra.pop(k)
            result3.append(extra)

        #for x in keys:
            #print(result[0][x])

        #print(result2)
        #print(result3)
        
        result['value']=result['errors']
        result4=[]
        result_dict=result.to_dict('records')
        for x in range(len(result_dict)):
            extra={k:result_dict[x][k] for k in keys }
            #for n in newkeys:
            #    extra[n] = extra.pop(k)
            result4.append(extra)
            
        errors_graph={'name':'Baseline error','value':base_mae},{'name':'Prediction error','value':pred_mae},{'name':'Improvement','value':base_mae-pred_mae}
        
#        baseerr_mean = {'name':'Base Errors Avg','series':[{'name':result['openingdate'].iloc[0],'value':base_mae},{'name':result['openingdate'].iloc[-1],'value':base_mae}]}
#        errors_mean = {'name':'Errors Avg','series':[{'name':result['openingdate'].iloc[0],'value':pred_mae},{'name':result['openingdate'].iloc[-1],'value':pred_mae}]}
        
        final = [{'name':'Issues','series':result3},{'name':'Prediction','series':result2},{'name':'Errors','series':result4},errors_graph,{'name':'Original issues','values':result.to_dict('records')}]

        

        return final
        #return jsonify(final)
        #return json.dumps(result.tolist())

def serialize(self):
        return {
            'name': self.gene_id, 
            'value': self.gene_symbol,
        }




def toDate(epoch):
    return datetime.datetime.fromtimestamp(epoch)
    

