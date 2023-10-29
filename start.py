from flask import Flask, request, jsonify
import pickle 
import numpy as np


app = Flask(__name__)
# model = pickle.load(open('Flask-Heroku-Deploy/random_forest_model_1_pk1' , 'rb'))

@app.route('/')
def home():
    return "Welcome Dude that's a dump API"
@app.route("/predict" , methods = ["GET"])
# def predict():
#     age = request.args.get('age')
#     sex = request.args.get('sex')
#     cp = request.args.get('cp')
#     trestbps = request.args.get('trestbps')
#     chol = request.args.get('chol')
#     fbs = request.args.get('fbs')
#     restecg = request.args.get('restecg')
#     thalach = request.args.get('thalach')
#     exang = request.args.get('exang')
#     oldpeak = request.args.get('oldpeak')
#     slope = request.args.get('slope')
#     ca = request.args.get('ca')
#     thal = request.args.get('thal')
#     makeprediction = model.predict([[age , sex , cp , trestbps ,
#                                       chol , fbs , restecg ,
#                                         thalach , exang , oldpeak ,
#                                           slope , ca , thal ]])
#     output = makeprediction.tolist()

#     return jsonify({"prediction" : list(output)})

    


if __name__ == "__main__":
    app.run(debug=True)
