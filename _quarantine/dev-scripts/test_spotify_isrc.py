import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

client_id = os.environ.get("SPOTIFY_CLIENT_ID")
if not client_id:
    print("NO SPOTIFY CREDENTIALS")
else:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
    res = sp.search(q="Get Lucky Daft Punk", type="track", limit=1)
    if res['tracks']['items']:
        track = res['tracks']['items'][0]
        print("ISRC:", track.get('external_ids', {}).get('isrc'))
