import urllib.request
import urllib.parse
import json
import base64
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Credentials
SPOTIFY_CID = "74611a31fcb848ee955ba8bb6d0920f2"
SPOTIFY_SECRET = "4b72c991cba9498385d4caf56fa96de6"
TIDAL_CID = "lmSad0LBn4aMT3LW"
TIDAL_SECRET = "DbHZBScmCwOjdrmCJlPo8DAznnQrYFjt0VeuAgZuH0k="

def get_tidal_token():
    auth_str = f"{TIDAL_CID}:{TIDAL_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    url = "https://auth.tidal.com/v1/oauth2/token"
    # Using client_credentials
    data = urllib.parse.urlencode({'grant_type': 'client_credentials'}).encode()
    req = urllib.request.Request(url, data=data)
    req.add_header('Authorization', f'Basic {b64_auth}')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())['access_token']

def test_flow():
    # 1. Spotify
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CID, client_secret=SPOTIFY_SECRET))
    res = sp.search(q="Daft Punk Get Lucky", type="track", limit=1)
    if not res['tracks']['items']:
        print("Spotify track not found")
        return
    track = res['tracks']['items'][0]
    isrc = track.get('external_ids', {}).get('isrc')
    print(f"Spotify ISRC for {track['name']}: {isrc}")
    
    if not isrc:
        return
        
    # 2. Tidal
    token = get_tidal_token()
    url = f"https://openapi.tidal.com/v2/tracks?countryCode=US&filter[isrc]={isrc}"
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('accept', 'application/vnd.tidal.v2+json')
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            items = data.get('data', [])
            if not items:
                print("No Tidal matching tracks")
            else:
                t = items[0]
                print(f"Tidal Match URL: {t.get('attributes', {}).get('url', 'N/A')}")
                print(f"Tidal Match ID: {t.get('id')}")
    except urllib.error.HTTPError as e:
        print(f"Tidal API HTTP Error {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"Tidal API error: {e}")

if __name__ == "__main__":
    test_flow()
