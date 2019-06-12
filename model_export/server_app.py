import flask
from flask import Flask, globals, request, g, jsonify
import json
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('elastic_regressor.pkl','rb'))
@app.route('/elastic_regressor',methods=['POST'])

def main():
    uploaded_files = globals.request.files.getlist('files')
    datafile = uploaded_files[0].read().decode('ascii')
    input_examples = [[float(i) for i in j.split(',')]  for j in datafile.split("\n") if j!='']
    results = {}
    results["predictions"] = list(model.predict( input_examples ))
    return  json.dumps(results, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)