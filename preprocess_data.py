import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Read the dataset
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "ex.csv")

df = pd.read_csv(csv_path)

df = df[['track_name', 'artists', 'album_name', 'track_genre', 'popularity']]
df = df.dropna()
df = df.drop_duplicates(subset='track_name')

# Perform data preprocessing
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df.sample(n=5000, random_state=42)

df['album_name'] = df['album_name'].str.replace(' ', '')
df['artists'] = df['artists'].str.replace(';', ' ', regex=False)

# Create tags
df['tags'] = (
    df['artists'] + " " +
    df['track_genre'] + " " +
    df['album_name'] + " " +
    df['popularity'].astype(str)
)

# Create feature vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags'])

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

# Create a copy of the DataFrame
new_df = df[['track_name', 'artists', 'track_genre', 'tags']].copy()

# Rename columns in the copied DataFrame
new_df.rename(columns={
    'track_name': 'title',
    'artists': 'artist',
    'track_genre': 'genre'
}, inplace=True)

new_df = new_df.reset_index(drop=True)

# Define recommendation function
def recommend(music):
    music_index = new_df[new_df['title'] == music].index[0]
    distances = similarity[music_index]
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in music_list:
        print(new_df.iloc[i[0]].title)

# Save trained model
os.makedirs('models', exist_ok=True)
pickle.dump(new_df, open('models/musicrec.pkl', 'wb'))
pickle.dump(similarity, open('models/similarities.pkl', 'wb'))