import os
from flask import Flask
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_restful import Resource, Api

from app import predictor



print('Initializing app...')

# Initialize application
app = Flask(__name__, static_folder=None)

#Initializa Bcrypt
bcrypt = Bcrypt(app)

# Enabling cors
CORS(app)

#app = Api(app)



# Initialize Flask MongoClient

#connection params
dbuser = 'portfoliouser'
dbpassword = 'portfolio13'
host = 'ds237475.mlab.com'
port =  37475
dbname = 'portfolio'
dbcollection = 'Projects'

#'mongodb://portfoliouser:portfolio13@ds237475.mlab.com:37475/portfolio',['Projects']
client = MongoClient(host,port)
db = client[dbname]
db.authenticate(dbuser,dbpassword)
collection = db[dbcollection]

print('Connected to',host)









from flask_jsonpify import jsonify


@app.route("/")
def hello():
    return jsonify({'text':'Hello World!'})

@app.route("/issues")
def issue():
    res = predictor.Issues()
    res.get()
    return jsonify(res.get())
    

@app.route("/preds")
def pred():
    res = predictor.Predictions()
    res.get()
    return jsonify(res.get())


    

#app.add_resource(predictor.Issues, '/issues') 
#app.add_resource(predictor.Predictions, '/preds') 