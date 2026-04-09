import requests
import base64
import json

TIDAL_CID = "lmSad0LBn4aMT3LW"
TIDAL_SECRET = "DbHZBScmCwOjdrmCJlPo8DAznnQrYFjt0VeuAgZuH0k="

def test():
    auth_str = f"{TIDAL_CID}:{TIDAL_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    url = "https://auth.tidal.com/v1/oauth2/token"
    res = requests.post(url, data={'grant_type': 'client_credentials'}, headers={
        'Authorization': f'Basic {b64_auth}'
    })
    token = res.json()['access_token']

    endpoints = [
        "https://openapi.tidal.com/v2/catalogs/tracks",
        "https://openapi.tidal.com/v2/catalog/tracks",
        "https://openapi.tidal.com/v1/catalogs/tracks",
        "https://openapi.tidal.com/v1/catalog/tracks"
    ]
    for url in endpoints:
        print(f"\nTrying: {url}")
        res = requests.get(url, params={'countryCode': 'US', 'filter[isrc]': 'USSM11302824'}, headers={
            'Authorization': f'Bearer {token}',
            'accept': 'application/vnd.tidal.v2+json'
        })
        print(f"Status: {res.status_code}")
        print(f"Body: {res.text[:100]}")

if __name__ == "__main__":
    test()
