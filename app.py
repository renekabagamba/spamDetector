from flask import Flask, render_template, request
import numpy as np
import joblib


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET','POST'])
def predict():

    if request.method == "POST":
        #get form data
        email_text = request.form.get('email_text')

        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(email_text)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass


def preprocessDataAndPredict(email_text):

    #convert value data into numpy array
    input_data = np.array([email_text])

    #reshape array
    input_data = input_data.reshape(1,-1)
    print(input_data)

    #open file
    file = open("spam_detector.pkl","rb")

    #load trained model
    trained_model = joblib.load(file)

    #predict
    prediction = trained_model.predict(input_data)

    return prediction



if __name__ == '__main__':
    port = process.env.PORT | 5000
    app.run(host="0.0.0.0", port=port)
