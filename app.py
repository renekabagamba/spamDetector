from flask import Flask, render_template, request
import numpy as np
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word


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
            return render_template('predict.html',
                                    email = email_text,
                                    prediction = prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass


def preprocessDataAndPredict(email_text):

    #pre processing steps like lower case, stemming and lemmatization
    email_text = email_text.lower()
    stop = stopwords.words('english')

    email_text = " ".join(x for x in email_text.split() if x not in stop)
    st = PorterStemmer()

    email_text = " ".join ([st.stem(word) for word in email_text.split()])
    email_text = " ".join ([Word(word).lemmatize() for word in email_text.split()])

    #open file
    file_model = open('spam_detector.pkl', "rb")
    file_tfidf_vect = open('tfidf.pkl', "rb")

    #load the trained model
    trained_model = joblib.load(file_model)
    tfidf_vect = joblib.load(file_tfidf_vect)

    new_email_tfidf =  tfidf_vect.transform([email_text])

    prediction = trained_model.predict(new_email_tfidf)

    return prediction



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv('PORT')))
