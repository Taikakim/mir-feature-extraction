import os
import requests
import base64

SPOTIFY_CID = "74611a31fcb848ee955ba8bb6d0920f2"
SPOTIFY_SECRET = "4b72c991cba9498385d4caf56fa96de6"
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

def test_tidal(isrc):
    token = get_tidal_token()
    # Explicitly string formatting the URL to prevent url-encoding the brackets
    url = f"https://openapi.tidal.com/v2/tracks?countryCode=US&filter[isrc]={isrc}"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'accept': 'application/vnd.tidal.v2+json'
    }
    
    res = requests.get(url, headers=headers)
    print(f"URL: {res.url}")
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text[:300]}")
    
    # Try finding the track URL natively through text search as another fallback
    url2 = f"https://openapi.tidal.com/v2/search?query={isrc}&type=TRACKS&countryCode=US"
    res2 = requests.get(url2, headers=headers)
    print(f"\nURL: {res2.url}")
    print(f"Status: {res2.status_code}")
    print(f"Response: {res2.text[:300]}")

if __name__ == "__main__":
    print("Testing Get Lucky (USQX91300108)")
    test_tidal("USQX91300108")
    print("\nTesting USSM11302824")
    test_tidal("USSM11302824")
