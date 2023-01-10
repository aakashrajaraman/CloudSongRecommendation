from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

from flask import Flask, render_template

app = Flask(__name__)
data = pd.read_csv('data.csv')
genre_data = pd.read_csv('data_by_genres.csv')
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

@app.route('/', methods = ['GET'])
def hello():
    return render_template('index.html')
@app.route('/', methods = ['POST'])
def run():
    song1 = request.form.get("song1")
    song2 = request.form.get("song2")
    song3 = request.form.get("song3")
    song4 = request.form.get("song4")
    song5 = request.form.get("song5")
    year1 = int(request.form.get('year1'))
    year2 = int(request.form.get('year2'))
    year3 = int(request.form.get('year3'))
    year4 = int(request.form.get('year4'))
    year5 = int(request.form.get('year5'))
    

    
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="31f0885e068a4540b6d788bba49c6be6",client_secret="674e7dc66a5e4ac8a34d5daaf4699799"))

    def find_song(name, year):
        song_data = defaultdict()
        results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = sp.audio_features(track_id)[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)

    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    def get_song_data(song, spotify_data):
        
        try:
            song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                    & (spotify_data['year'] == song['year'])].iloc[0]
            return song_data
        
        except IndexError:
            return find_song(song['name'], song['year'])
            
    def get_mean_vector(song_list, spotify_data):
        song_vectors = []
        
        for song in song_list:
            song_data = get_song_data(song, spotify_data)
            if song_data is None:
                print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
                continue
            song_vector = song_data[number_cols].values
            song_vectors.append(song_vector)  
        
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis=0)

    def flatten_dict_list(dict_list):
        
        flattened_dict = defaultdict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []
        
        for dictionary in dict_list:
            for key, value in dictionary.items():
                flattened_dict[key].append(value)
        return flattened_dict

    def recommend_songs( song_list, spotify_data, n_songs=10):
        
        metadata_cols = ['name', 'year', 'artists']
        song_dict = flatten_dict_list(song_list)
        
        song_center = get_mean_vector(song_list, spotify_data)
        scaler = song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(spotify_data[number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])
        
        rec_songs = spotify_data.iloc[index]
        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
        return rec_songs[metadata_cols].to_dict(orient='records')
    res= recommend_songs([{'name': song1, 'year':year1},
                {'name': song2, 'year': year2},
                {'name': song3, 'year': year3},
                {'name': song4, 'year': year4},
                {'name': song5, 'year': year5}],  data)
    return render_template('results.html', res = res, len = len(res))
    

if __name__ == '__main__':
    app.run()