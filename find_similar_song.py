import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

tracks = pd.read_csv("dataset.csv")

features = [
    "valence", "tempo", "energy", "danceability", "instrumentalness",
    "loudness", "acousticness", "speechiness", "liveness",
]
tracks = tracks.dropna(subset=features).reset_index(drop=True)

# Define weights for each feature - bigger weight means more importance in similarity calculation
weights = np.array([1.0, 0.7, 1.0, 1.0, 0.8, 0.5, 0.8, 0.4, 0.3])

scaler = StandardScaler()
X = scaler.fit_transform(tracks[features])
X = X * np.sqrt(weights)  # Apply weights to features

model = NearestNeighbors(metric="euclidean")
model.fit(X)


def get_similar_songs(song_name, artist):

    matches = tracks.index[
        (tracks["name"] == song_name)
        & tracks["artists_name"].str.contains(artist, na=False, regex=False)
    ]

    if len(matches) == 0:
        print("Song not found")
        return

    song_index = matches[0]
    excluded_indices = set(matches)

    n_neighbors = min(10 + len(excluded_indices), len(tracks))
    distances, positions = model.kneighbors(
        X[song_index].reshape(1, -1), n_neighbors=n_neighbors
    )

    results = [
        (position, distance)
        for position, distance in zip(positions[0], distances[0])
        if position not in excluded_indices
    ][:5]

    for position, distance in results:
        song = tracks.iloc[position]
        print(
            f"{song['name']} — {song['artists_name']} "
            f"({song['genres']}, distance: {distance:.2f})"
        )

#Test
#get_similar_songs("New Rules", "Dua Lipa")