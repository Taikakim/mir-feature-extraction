import urllib.request
import json
import base64

TIDAL_CID = "lmSad0LBn4aMT3LW"
TIDAL_SECRET = "DbHZBScmCwOjdrmCJlPo8DAznnQrYFjt0VeuAgZuH0k="

def get_tidal_token():
    auth_str = f"{TIDAL_CID}:{TIDAL_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    url = "https://auth.tidal.com/v1/oauth2/token"
    # To avoid importing urllib.parse for this
    data = b'grant_type=client_credentials'
    req = urllib.request.Request(url, data=data)
    req.add_header('Authorization', f'Basic {b64_auth}')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode())
        return res_data.get('access_token')

def search_tidal_by_isrc(token, isrc):
    # Using urllib.request with a hardcoded string prevents url-encoding of [ and ]
    url = f"https://openapi.tidal.com/v2/tracks?countryCode=US&filter[isrc]={isrc}"
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('accept', 'application/vnd.tidal.v2+json')
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            return res_data
    except Exception as e:
        print(f"Error for {isrc}: {e}")
        try:
            print("HTTP Status Code:", e.code)
            print("Body:", e.read().decode())
        except:
            pass
        return None

if __name__ == "__main__":
    token = get_tidal_token()
    print("Testing Get Lucky (USQX91300108)...")
    data1 = search_tidal_by_isrc(token, "USQX91300108")
    if data1: print("Success:", json.dumps(data1, indent=2)[:500])

    print("\nTesting Get Lucky 2 (USSM11302824)...")
    data2 = search_tidal_by_isrc(token, "USSM11302824")
    if data2: print("Success:", json.dumps(data2, indent=2)[:500])
