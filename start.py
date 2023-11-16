from flask import Flask, render_template, request ,jsonify
import pickle 
import numpy as np


app = Flask(__name__)
model = pickle.load(open('The_Medical_Model1.pkl' , 'rb'))


@app.route('/')
def home():
    return "Welcome Dude that's a dump API"
@app.route("/predict" , methods = ["GET"])
def predict():
    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')
    trestbps = request.args.get('trestbps')
    chol = request.args.get('chol')
    fbs = request.args.get('fbs')
    restecg = request.args.get('restecg')
    thalach = request.args.get('thalach')
    exang = request.args.get('exang')
    oldpeak = request.args.get('oldpeak')
    slope = request.args.get('slope')
    
    ca = request.args.get('ca')
    thal = request.args.get('thal')
    makeprediction = model.predict([[age , sex , cp , trestbps ,
                                      chol , fbs , restecg ,
                                        thalach , exang , oldpeak ,
                                          slope , ca , thal ]])
    makeprediction_prob = model.predict_proba([[age , sex , cp , trestbps ,
                                      chol , fbs , restecg ,
                                        thalach , exang , oldpeak ,
                                          slope , ca , thal ]])
    
    output = makeprediction.tolist()
    output_2 = makeprediction_prob.tolist()

    if output[0] == 1 :
        output = "positive"
    else :
        output = "negative"

    return jsonify({"Your result is " : output , 
                    #"No_probability" : output_2[0][0], 
                    #"Yes_probability" : output_2[0][1]
                    })
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = 1

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ == "__main__":
    app.run(debug=True)
