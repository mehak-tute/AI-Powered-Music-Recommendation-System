import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

# Load the preprocessed data
new_df = pickle.load(open('models/musicrec.pkl', 'rb'))
similarity = pickle.load(open('models/similarities.pkl', 'rb'))


# Function to get song details from Deezer (free, no API key needed)
def get_song_details(song_name):
    # Try Deezer first
    try:
        url = f"https://api.deezer.com/search?q={song_name}&limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('data') and len(data['data']) > 0:
            cover = data['data'][0]['album'].get('cover_big')
            if cover:
                return cover
    except Exception as e:
        print("Deezer error:", e)

    # Fallback: try iTunes
    try:
        url = f"https://itunes.apple.com/search?term={song_name}&limit=1&entity=song"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('results') and len(data['results']) > 0:
            cover = data['results'][0].get('artworkUrl100')
            if cover:
                # Get higher resolution version
                return cover.replace('100x100', '500x500')
    except Exception as e:
        print("iTunes error:", e)

    # Final fallback: reliable placeholder
    return "https://placehold.co/250x250/1DB954/white?text=No+Image"


# Define recommendation function
def recommend(song):
    song_index = new_df.reset_index(drop=True)
    song_index = song_index[song_index['title'] == song].index[0]
    distances = similarity[song_index]
    recommended_songs = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendations = []
    for i in recommended_songs:
        song_name = new_df.iloc[i[0]].title
        song_poster = get_song_details(song_name)
        recommendations.append({
    'song_name': song_name,
    'artist': new_df.iloc[i[0]].artist,
    'genre': new_df.iloc[i[0]].genre,
    'song_poster': song_poster
})
    return recommendations


# Streamlit UI
st.set_page_config(
    page_title="AI Music Recommendation System",
    page_icon="🎧",
    layout="wide"
)

with st.sidebar:
    st.header("Project Details")

    st.markdown("""
**Recommendation Technique**
- Content-Based Filtering

**Machine Learning**
- CountVectorizer
- Cosine Similarity

**Dataset**
- Spotify Tracks Dataset

**Songs**
- 5,000

**Developed by**
- Mehakpreet Kaur
- Bachelor of Computer Applications (BCA)
""")

st.title("🎧 AI-Powered Music Recommendation System")
st.markdown(
    "Discover similar songs using **content-based machine learning**."
)
st.divider()

st.caption(
    "Developed by Mehakpreet Kaur • BCA Project • AI-Powered Music Recommendation System"
)

selected_song = st.selectbox(
    "🎵 Search for a song",
    sorted(new_df['title'].unique()),
    help="Start typing to quickly find a song."
)

if st.button("✨ Find Similar Songs", use_container_width=True):
    with st.spinner("Finding similar songs... 🎧"):
        recommendations = recommend(selected_song)

    st.subheader("🎶 Recommended Songs")
    cols = st.columns(5)
    for i, song in enumerate(recommendations):
        with cols[i % 5]:
            if song['song_poster']:
                            st.image(song['song_poster'], width=220)
            else:
                 st.write("🎵 No image available")
            st.markdown(
            f"<div style='font-size:18px;font-weight:bold'>{song['song_name']}</div>",
            unsafe_allow_html=True
            )
            st.caption(f"🎤 {song['artist']}")
            st.caption(f"🎼 {song['genre'].title()}")
            