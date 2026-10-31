import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

tracks_df = pd.read_csv('dataset.csv')

# Scatter matrix to visualize the relationships between features
# reduced_df = tracks_df[:1000]
# pd.plotting.scatter_matrix(reduced_df[['valence', 'tempo', 'energy', 'danceability', 'instrumentalness']], figsize=(20, 12))
# plt.savefig("scatter_matrix.png")
# plt.show()

# Determine the optimal number of clusters using the elbow method and silhouette score
ks = range(2, 11)
inertias, sils = [], []
X =tracks_df[['valence', 'tempo', 'energy', 'danceability', 'instrumentalness']]
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertias, marker="o")
ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, sils, marker="o")
ax[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
plt.savefig("elbow_silhouette.png")

