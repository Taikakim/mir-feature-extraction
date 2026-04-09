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

def test_isrc(isrc):
    session = tidalapi.Session()
    # Mocking the session tokens using client_credentials
    # tidalapi normally expects oauth tokens containing users, but this might be enough for backend queries
    try:
        token = get_tidal_token()
        session.session_id = None
        session.access_token = token
        session.token_type = "Bearer"
        session.country_code = 'US'
        # The library uses these internally when making requests
    except Exception as e:
        print("Auth fallback failed:", e)

    try:
        tracks = session.get_tracks_by_isrc(isrc)
        print(f"\nISRC: {isrc}")
        if tracks:
            for t in tracks:
                print(f"Found: {getattr(t, 'name', 'N/A')} - {getattr(t, 'id', 'N/A')} - {t.get_url()}")
        else:
            print("No tracks found")
    except Exception as e:
        print("Search failed:", e)

if __name__ == "__main__":
    test_isrc("USQX91300108")
    test_isrc("USSM11302824")
