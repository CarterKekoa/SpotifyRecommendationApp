# we are going to use Flask, a micro web framework
import os
import pickle
from flask import Flask, jsonify, request, render_template
import importlib
import SpotifyAPIClient
import pprint
import csv
import random
from SpotifyAPIClient import SpotifyAPI
import mysklearn.myutils
import mysklearn.myutils as myutils
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 

# make a Flask app
app = Flask(__name__)

app.config["APP_DIR"] = os.path.dirname(os.path.abspath(__file__)) # absolute path to this file
app.config["APP_DATA"] = os.path.join(app.config["APP_DIR"], "data")
app.config["SPOTIFY_DATA"] = os.path.join(app.config["APP_DATA"], "track-audio-features-all.txt")


@app.route("/", methods=["POST", "GET"])
def index():
    track_data = MyPyTable().load_from_file(app.config["SPOTIFY_DATA"])
    attribute_values = []
    requires_format = ['danceability', 'energy', 'speechiness', 'valence', 'acousticness', 'instrumentalness', 'liveness']
    # client_id = 'd7a2e6f4a8434550baa5eda073f0a6a3'
    # client_secret = 'def46d47ba584e378b7667645666a468'
    # playlist_id = '7L736vCRhBe5EapwwkutUl'

    if request.method == "POST":
        try:
            if request.form['button'] == 'submitAttsButton':
                print("hello")
                attribute_values.append([request.form['sub-genre']])
                attribute_values.append([float(request.form['danceability'])])
                attribute_values.append([float(request.form['energy'])])
                attribute_values.append([float(request.form['loudness'])])
                attribute_values.append([float(request.form['speechiness'])])
                attribute_values.append([float(request.form['tempo'])])
                attribute_values.append([float(request.form['valence'])])
                print(attribute_values)
            else:
                print("hello2")
        except:
            print("bad")
    
    return render_template('main.html', atts=attribute_values)




# we need to add two routes (functions that handle requests)
# one for the homepage
@app.route("/old", methods=["POST", "GET"])
def old():
    track_data = MyPyTable().load_from_file(app.config["SPOTIFY_DATA"])
    chosen_attributes = []
    requires_format = ['danceability', 'energy', 'speechiness', 'valence', 'acousticness', 'instrumentalness', 'liveness']
    chosen_ones = []
    # client_id = 'd7a2e6f4a8434550baa5eda073f0a6a3'
    # client_secret = 'def46d47ba584e378b7667645666a468'
    # playlist_id = '7L736vCRhBe5EapwwkutUl'

    if request.method == "POST":
        try:
            if request.form['button'] == 'submitAttsButton':
                chosen_attributes = request.form.getlist("attribute")
                for att in chosen_attributes:
                    print(att)
                    if att in requires_format:
                        chosen_ones.append(myutils.format_num(track_data.get_column(att)))
                    else:
                        chosen_ones.append(track_data.get_column(att))
                print(chosen_ones)

                # dance_bins = myutils.bin_vals(danceability)
                # energy_bins = myutils.bin_vals(energy)
                # loudness_bins = myutils.bin_loudness(loudness)
                # speechiness_bins = myutils.bin_vals(speechiness)
                # tempo_bins = myutils.bin_tempo(tempo)
                # valence_bins = myutils.bin_vals(valence)

                # dance_bin_count = [[len(dance_bins[0])],[len(dance_bins[1])],[len(dance_bins[2])],[len(dance_bins[3])],[len(dance_bins[4])],[len(dance_bins[5])],[len(dance_bins[6])],[len(dance_bins[7])],[len(dance_bins[8])],[len(dance_bins[9])]]


                # x_vals = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100']
                # y_vals = []
                # for i in range(len(dance_bin_count)):
                #     y_vals.append(dance_bin_count[i][0])
        except:
            print("bad")

    return render_template('old.html', chosen_atts=chosen_attributes, attributes=track_data.column_names)

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host="0.0.0.0", port=port) # TODO: set debug to False for production
    # by default, Flask runs on port 5000