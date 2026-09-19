# AI-Powered Music Recommendation System

This project is an AI-powered music recommendation system developed using Python and Streamlit. It uses content-based filtering to recommend the top 5 songs similar to a user-selected track.

## Project Overview

The system uses song metadata such as artist, genre, album, and popularity to generate recommendations. CountVectorizer converts the combined text features into numerical vectors, and Cosine Similarity is used to compare songs and identify the most similar tracks.

## Features

- Searchable song selection interface
- Top 5 similar song recommendations
- Content-based filtering
- Metadata-based similarity calculation
- Album artwork retrieval
- Interactive Streamlit interface

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CountVectorizer
- Cosine Similarity
- Streamlit
- Deezer API
- iTunes Search API
- Pickle

## Dataset

The project uses the Spotify Tracks Dataset containing approximately 114,000 tracks.

For the current implementation, the following attributes are used for recommendation:

- Track Name
- Artist
- Album
- Genre
- Popularity

A working subset of 5,000 tracks is used for efficient similarity calculation.

## Recommendation Process

1. User selects a song.
2. Song metadata is converted into a numerical representation using CountVectorizer.
3. Cosine Similarity calculates similarity with other songs.
4. Songs are ranked according to their similarity scores.
5. The top 5 similar songs are displayed.

## Application

The application is developed using Streamlit and displays the recommended song name, artist, genre, and album artwork.

## Project Structure

AI-Music-Recommendation-System/
├── app.py
├── preprocess_data.py
└── README.md

## Limitations

* Uses a limited working subset of the complete dataset.
* Recommendations do not use user listening history.
* The current implementation does not use audio features for similarity.
* Album artwork retrieval requires an internet connection.
* The application currently runs locally.

## Future Scope

* Incorporate audio features such as danceability, energy, tempo, and valence.
* Add user preferences and listening history.
* Improve recommendation accuracy using advanced techniques.
* Deploy the application online.
* Integrate real-time music data.

## References

* Spotify Tracks Dataset (https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
* Python Documentation (https://docs.python.org/3/)
* Scikit-learn Documentation (https://scikit-learn.org/stable/)
* Pandas Documentation (https://pandas.pydata.org/docs/)
* Streamlit Documentation (https://docs.streamlit.io/)
* Deezer API Documentation (https://developers.deezer.com/api)
* iTunes Search API (https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/)
