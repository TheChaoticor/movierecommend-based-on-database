import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_content_based_recommendations(movie_title, movies_df, cosine_sim, top_n=5):
    """
    Recommends movies similar to the given movie based on metadata.

    Args:
        movie_title (str): Title of the movie.
        movies_df (pd.DataFrame): DataFrame containing movie metadata.
        cosine_sim (np.ndarray): Precomputed cosine similarity matrix.
        top_n (int): Number of recommendations to return.

    Returns:
        List of dictionaries containing recommended movie titles and poster links.
    """
    try:
        # Get the index of the input movie
        movie_idx = movies_df[movies_df['Series_Title'] == movie_title].index[0]
    except IndexError:
        return {"error": "Movie not found. Please check the title."}

    # Get similarity scores for all movies
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    
    # Sort movies by similarity scores
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N recommended movies (excluding the input movie itself)
    recommended_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    
    # Fetch titles and poster links
    recommendations = [
        {
            "title": movies_df.iloc[i]["Series_Title"],
            "poster_link": movies_df.iloc[i]["Poster_Link"]
        }
        for i in recommended_indices
    ]
    
    return recommendations

def update_cosine_similarity(movies_df):
    """
    Updates the cosine similarity matrix based on the movie descriptions.

    Args:
        movies_df (pd.DataFrame): DataFrame containing movie metadata.

    Returns:
        np.ndarray: Updated cosine similarity matrix.
    """
    # Assuming you have a 'Description' column or similar for content-based filtering
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['Description'])  # Adjust based on your DataFrame structure
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_movies(movie_title, movies_df, cosine_sim, top_n=5):
    """
    Recommends movies similar to the given movie.
    
    Args:
        movie_title (str): Title of the movie.
        movies_df (pd.DataFrame): DataFrame containing movie metadata.
        cosine_sim (np.ndarray): Precomputed cosine similarity matrix.
        top_n (int): Number of recommendations to return.

    Returns:
        List of dictionaries containing recommended movie titles and poster links.
    """
    try:
        # Find the index of the movie in the DataFrame
        idx = movies_df[movies_df['Series_Title'] == movie_title].index[0]
    except IndexError:
        return {"error": "Movie not found. Please check the title."}
    
    # Get pairwise similarity scores for this movie
    similarity_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top_n similar movies (excluding the input movie itself)
    top_movies = sorted_scores[1:top_n + 1]
    
    # Create the recommendations list
    recommendations = []
    for i, score in top_movies:
        recommendations.append({
            "title": movies_df.iloc[i]['Series_Title'],
            "poster_link": movies_df.iloc[i]['Poster_Link']
        })
    
    # Return the recommendations in the desired format
    return {"content_based": recommendations}