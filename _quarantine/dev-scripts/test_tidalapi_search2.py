import tidalapi
import requests
import base64
import os

TIDAL_CID = "lmSad0LBn4aMT3LW"
TIDAL_SECRET = "DbHZBScmCwOjdrmCJlPo8DAznnQrYFjt0VeuAgZuH0k="

def get_tidal_token():
    auth_str = f"{TIDAL_CID}:{TIDAL_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    url = "https://auth.tidal.com/v1/oauth2/token"
    res = requests.post(url, data={'grant_type': 'client_credentials'}, headers={
        'Authorization': f'Basic {b64_auth}'
    })
    return res.json()['access_token']

def test():
    session = tidalapi.Session()
    try:
        token = get_tidal_token()
        session.session_id = None
        session.access_token = token
        session.token_type = "Bearer"
        session.country_code = 'US'
    except Exception as e:
        print("Auth fallback failed:", e)

    try:
        tracks = session.search('Daft Punk Get Lucky', models=[tidalapi.Track])
        if 'tracks' in tracks and tracks['tracks']:
            t = tracks['tracks'][0]
            print(f"Name: {t.name}")
            print(f"ID: {t.id}")
            print(f"ISRC: {t.isrc}")
            print(f"URL: {t.get_url()}")
        else:
            print("No tracks found")
    except Exception as e:
        print("Search failed:", e)

if __name__ == "__main__":
    test()
