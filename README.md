# SpotifyRecommendationApp
**Developed by Armando Valdez & Carter Mooring**

This is a Flask App deployed on [Heroku](https://hub.docker.com/r/continuumio/anaconda3) with Docker that makes use of a Spotify song dataset with 5000 values to train on. A user can send a 
request to our endpoint or interact with our web app to enter song attribute values and predict that tracks popularity.

*Find the Deployed Version Here: https://spotify-popularity-classifier.herokuapp.com*

Originially our app would have trained using a users Spotify playlist data and recomend new songs based on that users song data in the playlist. However the Spotify
API limited the data we could access from a users account, so we were unable to access a users personal song popularity value. This value is used by Spotify to 
produce their New Music Friday recommendations. As a result we used a pre-set dataset.

## Data Set
  Our data set has 5000 instances of songs that each contained 23 attributes:  
  `track_id, track_name, track_artist, track_popularity, track_album_id, track_album_name, track_album_release_date, playlist_name, 
  playlist_id, playlist_genre, playlist_subgenre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness,
  liveness, valence, tempo, duration_ms`

  One song instace example:
  `6f807x0ima9a1j3VPbc7VN, I Don't Care (with Justin Bieber) - Loud Luxury Remix, Ed Sheeran, 66, 2oCs0DGTsRO98Gh5ZSl2Cx, I Don't Care (with Justin Bieber) [Loud Luxury Remix], 
  2019-06-14, Pop Remix, 37i9dQZF1DXcZDD7cfEKhW, pop, dance pop, 0.748, 0.916, 6, -2.634, 1, 0.0583, 0.102, 0, 0.0653, 0.518, 122.036, 194754`
  
  From the instance above you can see the values get very specific, so to avoid overfitting we created bins to categorize the values most commonly on a scale of 1-10.
  This also helped yield better prediction results.
  
  Our Classifiers focused on just 7 of the available attributes that were determined as most useful based on the Spotify API documentation:  
  `playlist_subgenre, danceability, energy, loudness, speechiness, valence, tempo`

  The `playlist_subgenre` we use to train on has only 4 genres in our dataset: `dance pop, post-teen pop, electropop, indie poptimism`
  Obviously, these are not all the genres in music, meaning if we had more data and genres then we could more accurately predict popularity from given values. 
  Otherwise the values we are currently given could be values not normally seen in these genres and thus produce incorrect results for popularity.
  
  In the future it may be more beneficial to test different combinations of attributes to train on and find which attributes were able to predict more 
  accurately together.
  
## How to run the project
Our project is contained in a Docker Container that has all the packages we used to produce our result. This docker container is also used to deploy the app to Heroku.

#### **Local Host**
To run the app on local host you can open a terminal, navigate to the project directory, and run:
```
python recommendation_app.py
```
Then visit: http://localhost:5000/ to interact with the web app.

#### **Deployed (Prefered Method)**
Our web app is also deployed on Heroku and can be visited at: https://spotify-popularity-classifier.herokuapp.com

#### **Interaction**
* On the web app landing page there are text boxes asking you to input bin values (usually 1-10) to predict on. Clicking the submit button once you have entered 
all the values will cause the page to return the predicted popularity of the songs values you entered.
* You could also pass the attribute values through https://spotify-popularity-classifier.herokuapp.com/predict  
  * As an example you can try this url: https://spotify-popularity-classifier.herokuapp.com/predict?sub-genre=electropop&danceability=7&energy=5&loudness=3&speechiness=2&tempo=2&valence=4
* We are also currently developing the Spotify API portion so that you can have the option to search a specific song that we can automatically grab the attribute values for.

## Organization
```
SpotifyRecommendationApp
|
|-- data
|    |-- rules.p (pickleizer for the fit classifier data, allows us to make predictions on the web app)
|    |-- spotify-songs.txt (the storage of a users playlist using the Spotify API)
|    |-- track_audio_features_all.txt (current 5000 instance dataset of Spotify songs)
|
|-- mysklearn
|    |-- myclassifier.py (Naive Bayes, Decision Tree, & Random Forrest Classifiers)
|    |-- myevaluation.py (Helper functions for Classifiers)
|    |-- mypytable.py (data alteration helper)
|    |-- myutils.py (general helper functions)
|    |-- ploy_utils.py (helper functions for Jupyter notebook plotting)
|
|-- static
|    |-- main.css 
|
|-- templates
|    |-- base.html (overall html for the web app that all other html will inherit from)
|    |-- main.html (Contains the design for the user interactive landing page)
|
|-- tools
|    |-- Spotify_pickler.py (puts the fit data models into rules.p)
|    |-- SpotifyAPIClient.py (Spotify API connection and helper functions)
|
|-- Dockerfile (File used to deploy web app to Heroku)
|-- EDA.ipynb (Jupyter notebook of the EDA for our Classifiers on our data)
|-- FinalReport.ipynb (Final Jupyter class report on hte project)
|-- heroku.yml (Executes when web app is deployed)
|-- ProjectProposal.ipynb (Jupyter Notebook of our original project proposal)
|-- recommendation_app.py (the main web app file that is ran)
|-- test_myRandomForest.py (Test cases for the My Random Forest Classifier)
```



