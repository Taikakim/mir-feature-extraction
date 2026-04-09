import urllib.request
import urllib.parse
import json
import base64
import os
import sys

CLIENT_ID = "lmSad0LBn4aMT3LW"
CLIENT_SECRET = "DbHZBScmCwOjdrmCJlPo8DAznnQrYFjt0VeuAgZuH0k="

def get_tidal_token():
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    url = "https://auth.tidal.com/v1/oauth2/token"
    data = urllib.parse.urlencode({'grant_type': 'client_credentials'}).encode()
    req = urllib.request.Request(url, data=data)
    req.add_header('Authorization', f'Basic {b64_auth}')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())['access_token']

if __name__ == "__main__":
    try:
        token = get_tidal_token()
        print(f"Token: {token[:10]}...")
        
        # Try finding tracks by ISRC
        isrc = "USSM11302824" 
        endpoints = [
            f"https://openapi.tidal.com/tracks?countryCode=US&filter[isrc]={isrc}",
            f"https://openapi.tidal.com/v2/tracks?countryCode=US&filter[isrc]={isrc}",
            f"https://api.tidal.com/v1/tracks?filter[isrc]={isrc}&countryCode=US",
            f"https://openapi.tidal.com/search?query=USSM11302824&type=TRACKS&countryCode=US",
            f"https://openapi.tidal.com/v2/search?query=USSM11302824&type=TRACKS&countryCode=US"
        ]
        
        for url in endpoints:
            print(f"\nTrying: {url}")
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {token}')
            try:
                with urllib.request.urlopen(req) as response:
                    print("Status:", response.status)
                    print(response.read().decode()[:200])
            except urllib.error.HTTPError as e:
                print("HTTP Error:", e.code)
                print(e.read().decode()[:200])
            except Exception as e:
                print("Error:", e)
    except Exception as e:
        print("Fatal API error:", e)
