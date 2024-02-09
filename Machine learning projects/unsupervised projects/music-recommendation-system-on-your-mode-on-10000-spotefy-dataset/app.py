from flask import Flask, jsonify, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

app = Flask(__name__)

df = pd.read_csv(r"datasettt.csv")

# Selecting relevant features for clustering
features_for_clustering = df[['Energy', 'Valence', 'Acousticness', 'Instrumentalness']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
imputed_features = imputer.fit_transform(scaled_features)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(imputed_features)
    inertia.append(kmeans.inertia_)

# Based on the elbow curve, choose the optimal number of clusters
# Let's say k=5
k = 5

# Perform K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(imputed_features)

# Add the 'Cluster' column to the DataFrame
df['Cluster'] = clusters


# Function to recommend similar songs based on track features
def recommend_similar_songs(mode, top_n=5):
    song_features = df[df['Danceability'] == mode][['Energy', 'Valence', 'Acousticness', 'Instrumentalness']]
    if len(song_features) == 0:
        return []

    # Calculate the Euclidean distance between the selected song and all other songs
    distances = np.sqrt(np.sum((features_for_clustering - song_features.values[0])**2, axis=1))

    # Get the indices of the most similar songs
    similar_song_indices = np.argsort(distances)[1:top_n + 1]

    # Get the recommended similar songs
    recommended_songs = df.iloc[similar_song_indices]['Track Name'].tolist()

    return recommended_songs


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend/<float:mode>/<int:top_n>', methods=['GET'])
def recommend(mode, top_n):
    recommended_songs = recommend_similar_songs(mode, top_n)
    return jsonify({'recommended_songs': recommended_songs})


if __name__ == '__main__':
    app.run()