# we are going to use Flask, a micro web framework
import os
import pickle
from flask import Flask, jsonify, request, render_template
import importlib
import SpotifyAPIClient
import pprint
import csv
import random
import utils
from SpotifyAPIClient import SpotifyAPI
import mysklearn.myutils
import mysklearn.myutils as myutils
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyNaiveBayesClassifier
import mysklearn.myevaluation as myevaluation

# make a Flask app
app = Flask(__name__)

app.config["APP_DIR"] = os.path.dirname(os.path.abspath(__file__)) # absolute path to this file
app.config["APP_DATA"] = os.path.join(app.config["APP_DIR"], "data")
app.config["SPOTIFY_DATA"] = os.path.join(app.config["APP_DATA"], "track-audio-features-all.txt")
app.config["PICKLE_DATA"] = os.path.join(app.config["APP_DATA"], "rules.p")


@app.route("/", methods=["POST", "GET"])
def index():
    predicted = None
    attribute_values = []
    predict_set=[]
    requires_format = ['danceability', 'energy', 'speechiness', 'valence', 'acousticness', 'instrumentalness', 'liveness']
    # client_id = 'd7a2e6f4a8434550baa5eda073f0a6a3'
    # client_secret = 'def46d47ba584e378b7667645666a468'
    # playlist_id = '7L736vCRhBe5EapwwkutUl'

    if request.method == "POST":
        try:
            if request.form['button'] == 'submitAttsButton':
                print("hello")
                attribute_values.append(request.form['sub-genre'])
                attribute_values.append(float(request.form['danceability']))
                attribute_values.append(float(request.form['energy']))
                attribute_values.append(float(request.form['loudness']))
                attribute_values.append(float(request.form['speechiness']))
                attribute_values.append(float(request.form['tempo']))
                attribute_values.append(float(request.form['valence']))

                print(attribute_values)
                predict_set.append(attribute_values)
                predicted = predict_well(predict_set)
                print("predicted: ", predicted)
            else:
                print("helloalso")
                
        except:
            print("bad")
            utils.PrintException()
    
    return render_template('main.html', atts=attribute_values, prediction=predicted)

# one for the /predict 
@app.route("/predict", methods=["GET"])
def predict():
    # goal is to extract the 4 attribute values from query string
    # use the request.args dictionary
    # test url = "https://interview-flask-app.herokuapp.com/predict?sub-genre=electropop&danceability=7&energy=5&loudness=3&speechiness=2&tempo=2&valence=4"
        
    sub_genre = request.args.get("sub-genre", "") # check for the key, and the default
    danceability = int(request.args.get("danceability", ""))
    energy = int(request.args.get("energy", ""))
    loudness = int(request.args.get("loudness", ""))
    speechiness = int(request.args.get("speechiness", ""))
    tempo = int(request.args.get("tempo", ""))
    valence = int(request.args.get("valence", ""))
    print("Endpoint Vals:", sub_genre, danceability, energy, loudness, speechiness, tempo, valence)

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response

    # TODO: fix the hardcoding
    prediction = predict_well([[sub_genre, danceability, energy, loudness, speechiness, tempo, valence]]) # if anything goes wrong, this function will return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        # failure!!
        return "Error making prediction", 400

# this is the predict for pa6 my decision tree classifier
def predict_well(instance):
    # 1. we need a tree (and its header) to make a prediction
    #   we need to save a trained model (fit()) to a file
    #   so we can load that file into memory in another python
    #   process as a python object (predict())
    # import pickle and load the header and interview tree
    #   as Python objects we can use for step 2
    infile = open(app.config["PICKLE_DATA"], "rb") # r for read, b for binary
    priors, posteriors = pickle.load(infile)
    infile.close()

    # 2. use the posteriors to make a prediction
    try:
        # test url = "https://interview-flask-app.herokuapp.com/predict?level=Junior&language=Java&tweets=yes&phd=no"
        return nb_predict(priors, posteriors, instance) # recursive function
    except:
        return None

def nb_predict(priors, posteriors, instance):
    y_predicted = []
    for test in instance:
        probs = myutils.compute_probs(test, priors, posteriors)
        prediction = myutils.predict_from(probs)
        y_predicted.append(prediction)
    return y_predicted

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host="0.0.0.0", port=port) # TODO: set debug to False for production
    # by default, Flask runs on port 5000