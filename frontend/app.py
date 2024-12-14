import streamlit as st
import requests

st.title("Movie Recommendation System")

movie_title = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    if movie_title:
        response = requests.post('http://127.0.0.1:5000/recommend', json={'movie_title': movie_title})
        recommendations = response.json()
        
        # Print the recommendations to check the structure
        print(recommendations)  # This will show in the terminal

        # Check if the response contains an error
        if 'error' in recommendations:
            st.error(recommendations['error'])
        else:
            st.subheader("Recommended Movies:")
            for movie in recommendations:  # Change this if recommendations is a list
                st.write(f"**{movie['title']}**")
                st.image(movie['poster_link'])
    else:
        st.warning("Please enter a movie title.")