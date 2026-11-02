import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

tracks_df = pd.read_csv('dataset.csv')

model = KMeans(n_clusters=5)

features = ['valence', 'tempo', 'energy', 'danceability', 'instrumentalness']
tracks_df = pd.read_csv('dataset.csv').dropna(subset=features).reset_index(drop=True)
X = StandardScaler().fit_transform(tracks_df[features])
model.fit(X)

tracks_df['type'] = model.labels_
tracks_df.to_csv('result.csv')
