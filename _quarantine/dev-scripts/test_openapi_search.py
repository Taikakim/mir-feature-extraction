import requests
import base64
import json

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

if __name__ == "__main__":
    token = get_tidal_token()
    
    # Let's try to search via openapi
    url = "https://openapi.tidal.com/search"
    params = {
        'query': 'Daft Punk Get Lucky',
        'type': 'TRACKS',
        'countryCode': 'US'
    }
    headers = {
        'Authorization': f'Bearer {token}',
        'accept': 'application/vnd.tidal.v1+json'
    }
    
    res = requests.get(url, params=params, headers=headers)
    print("Status:", res.status_code)
    try:
        print(json.dumps(res.json(), indent=2)[:500])
    except:
        print(res.text)
