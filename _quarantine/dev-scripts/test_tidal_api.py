import urllib.request
import urllib.parse
import json
import base64
import os
import sys

# Tidal credentials from the user's file
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
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            return res_data.get('access_token')
    except Exception as e:
        print(f"Auth error: {e}")
        return None

def search_tidal_by_isrc(token, isrc):
    # Tidal openapi tracks endpoint
    # E.g., https://openapi.tidal.com/tracks?filter[isrc]=USSM11302824
    url = f"https://openapi.tidal.com/tracks?countryCode=US&filter[isrc]={isrc}"
    
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('accept', 'application/vnd.tidal.v1+json')
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            return res_data
    except Exception as e:
        print(f"Search error: {e}")
        return None

if __name__ == "__main__":
    token = get_tidal_token()
    if not token:
        print("Failed to get token")
        sys.exit(1)
        
    print(f"Got token: {token[:10]}...")
    
    # "Get Lucky" ISRC
    test_isrc = "USSM11302824"
    print(f"Searching for ISRC: {test_isrc}")
    
    data = search_tidal_by_isrc(token, test_isrc)
    if data:
        # data structure usually contains 'data' array
        items = data.get('data', [])
        if items:
            track = items[0]
            track_id = track.get('id')
            # Extract sharing URLs from attributes if available
            attributes = track.get('attributes', {})
            title = attributes.get('title')
            print(f"Found Track: {title} (ID: {track_id})")
            print(json.dumps(track, indent=2))
        else:
            print("No tracks found for that ISRC")
    else:
        print("Search request failed")
