import urllib.request
import json
import time

try:
    print("Testing API /tracks...")
    req = urllib.request.urlopen("http://127.0.0.1:7892/api/tracks", timeout=2)
    data = json.loads(req.read().decode())
    print(f"Tracks count: {len(data['tracks'])}")
except Exception as e:
    print(f"Error tracks: {e}")
