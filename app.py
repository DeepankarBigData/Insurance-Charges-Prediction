from flask import Flask, render_template, request
import pickle
import pandas as pd
from model import model,mod
app = Flask(__name__)

# Load the trained model
# model = pickle.load(open("mod.pkl", "rb"))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']

    sex = request.form['sex']
    sex = 1 if sex.lower() == 'male' else 0

    bmi = request.form['bmi']

    children = request.form['children']

    smoker = request.form['smoker']
    smoker = 1 if smoker.lower() == 'yes' else 0

    input_data = [
        age,
        sex,
        bmi,
        children,
        smoker]



    inputs = [float(i) for i in input_data]

    # Make the prediction using the loaded model
    prediction = mod.predict([inputs])

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True,host ='0.0.0.0')
