import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    in_features = [float(x) for x in request.form.values()]
    final_features = [np.array(in_features)]
    prediction = model.predict(final_features)
    output = 'good'
    if round(prediction[0],2)>0.5:
        output = 'good'
    else:
        output = 'bad'
    # output=round(prediction[0],2)
    return render_template('index.html',prediction_text='The Wine Quality should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)