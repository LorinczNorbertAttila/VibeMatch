## join artist genre information with track dataset
# drop irrelevant columns
import pandas as pd

album_df = pd.read_csv("spotify_albums.csv")
artist_df = pd.read_csv("spotify_artists.csv")
track_df = pd.read_csv("spotify_tracks.csv")

album = album_df.rename(columns={'id':"album_id"}).set_index('album_id')
artist = artist_df.rename(columns={'id':"artists_id",'name':"artists_name"}).set_index('artists_id')
track = track_df.set_index('album_id').join(album['release_date'], on='album_id' )
track.artists_id = track.artists_id.apply(lambda x: x[2:-2])
track = track.set_index('artists_id').join(artist[['artists_name','genres']], on='artists_id' )
track.reset_index(drop=False, inplace=True)
track.drop(columns = ['Unnamed: 0','country','track_name_prev','track_number','type'], inplace = True)

#preprocessing the uri column
track["uri"] = track["uri"].str.replace("spotify:track:", "")
track = track.drop(columns=['analysis_url', 'available_markets'])

track.to_csv("dataset.csv", index=False)