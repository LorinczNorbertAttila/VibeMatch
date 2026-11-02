import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

tracks_df = pd.read_csv('dataset.csv')

# Scatter matrix to visualize the relationships between features
# reduced_df = tracks_df[:1000]
# pd.plotting.scatter_matrix(reduced_df[['valence', 'tempo', 'energy', 'danceability', 'instrumentalness']], figsize=(20, 12))
# plt.savefig("scatter_matrix.png")
# plt.show()

# Determine the optimal number of clusters using the elbow method and silhouette score
features = ['valence', 'tempo', 'energy', 'danceability', 'instrumentalness']
X = StandardScaler().fit_transform(tracks_df[features].dropna()) 
# Standardize the audio features with StandardScaler so that `tempo` (~60-200) no longer dominates the distance calculation over the 0-1 features (valence, energy, danceability, instrumentalness).

ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)
    # sample_size speeds up the silhouette score calculation for large datasets
    sils.append(silhouette_score(X, km.labels_, sample_size=10000, random_state=42))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertias, marker="o")
ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, sils, marker="o")
ax[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
plt.savefig("elbow_silhouette.png")

