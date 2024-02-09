# Music Recommendation System using Flask

This project is a music recommendation system developed using Flask, a Python web framework. The system analyzes a dataset of music tracks and recommends similar songs based on specific features such as energy, valence, acousticness, and instrumentalness.

## Project Structure

- <h3 style="font-size: 1.2em;">`app.py`</h3>: The main Flask application file that defines the web server and routes.
- <h3 style="font-size: 1.2em;">`index.html`</h3>: The HTML template file for the main page.
- <h3 style="font-size: 1.2em;">`datasettt.csv`</h3>: The dataset containing information about various music tracks.

## Feature Selection and Recommendation Algorithm

The core of the recommendation system involves selecting relevant features for similarity and using a custom algorithm to recommend similar songs. The steps include:

1. **Data Loading**: Load the music dataset (`datasettt.csv`) using Pandas.
2. **Feature Selection**: Select relevant features for similarity, such as energy, valence, acousticness, and instrumentalness.
3. **Data Preprocessing**: Standardize the selected features and handle any missing values.
4. **Recommendation Algorithm**: The system includes a function `recommend_similar_songs` that takes a mode (danceability) and returns a list of recommended similar songs. The process may involve calculating distances or applying a custom similarity metric.

## Web Interface

The Flask web application provides a simple interface with an HTML template (`index.html`). The main page is accessible at the root URL ("/") and serves as a starting point. Additionally, there is an API endpoint (`/recommend/<float:mode>/<int:top_n>`) that accepts parameters for danceability mode and the number of top recommended songs.

## Usage

To run the application, execute the `app.py` script. The web server will start, and you can access the main page in your browser. To get song recommendations programmatically, use the `/recommend/<float:mode>/<int:top_n>` API endpoint.
