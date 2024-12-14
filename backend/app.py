from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
from model import get_content_based_recommendations  # Import your recommendation function

app = Flask(__name__)

# Load your movies DataFrame and cosine similarity matrix
movies_df = pd.read_csv('imdb_top_1000.csv')  # Update with your actual path
cosine_sim = np.load('cosine_similarity_matrix.npy')  # Update with your actual path

def get_movie_metadata(movie_title):
    # Example API call to fetch movie metadata
    response = requests.get(f"http://www.omdbapi.com/?t={movie_title}&apikey=YOUR_API_KEY")
    if response.status_code == 200:
        return response.json()
    return None

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_title = data.get('movie_title')
    
    # Try to get recommendations from the dataset
    recommendations = get_content_based_recommendations(movie_title, movies_df, cosine_sim)
    
    # If recommendations are empty or an error occurred, try to fetch from the API
    if 'error' in recommendations:
        movie_metadata = get_movie_metadata(movie_title)
        if movie_metadata and movie_metadata.get('Response') == 'True':
            # Create a temporary DataFrame for the new movie
            temp_movie = pd.DataFrame({
                'Series_Title': [movie_metadata['Title']],
                'Poster_Link': [movie_metadata.get('Poster', '')],
                # Add other relevant fields if necessary
            })
            # Combine with the existing DataFrame
            combined_df = pd.concat([movies_df, temp_movie], ignore_index=True)
            # Recompute cosine similarity with the new movie included
            # You may need to recalculate the cosine similarity matrix here
            # For simplicity, we will assume cosine_sim is already updated
            recommendations = get_content_based_recommendations(movie_metadata['Title'], combined_df, cosine_sim)
        else:
            return jsonify({"error": "Movie not found. Please check the title."})

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)