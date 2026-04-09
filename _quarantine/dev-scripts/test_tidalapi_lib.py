import tidalapi
import os

session = tidalapi.Session()
# We can login using the client credentials we have? 
# Wait, let's see if it works without login first for searching.
try:
    tracks = session.search('track', 'USQX91300108')
    print("Search without auth:", tracks)
except Exception as e:
    print("Search without auth failed:", e)

# How to use Client ID / Secret in tidalapi?
# It actually has `login_client_credentials()`:
try:
    TIDAL_CID = "lmSad0LBn4aMT3LW"
    TIDAL_SECRET = "DbHZBScmCwOjdrmCJlPo8DAznnQrYFjt0VeuAgZuH0k="
    session.login_api_key(TIDAL_CID, TIDAL_SECRET)
    print("Login API Key OK")
except Exception as e:
    print("login_api_key exception:", e)

try:
    session.login_client_credentials(TIDAL_CID, TIDAL_SECRET)
    print("Login client credentials OK")
except Exception as e:
    print("login_client_credentials exception:", e)

# Try fetching track by ID or searching
tracks = session.search('track', 'Daft Punk Get Lucky')
if 'tracks' in tracks and tracks['tracks']:
    t = tracks['tracks'][0]
    print(f"Found track: {t.name} - URL: {t.get_url()}")
    print(f"ISRC might be: {getattr(t, 'isrc', 'Missing')}")
else:
    print("No tracks found via text search.")
