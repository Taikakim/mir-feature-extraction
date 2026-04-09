import urllib.request
import json
import urllib.parse
# Let's try to query MusicBrainz to see if it exposes Tidal links for a known ID
# MBID: e.g. we need one
mbid = '6b12a14e-ebdb-4fc6-b7ff-1f558b8d141e' # random recording
url = f'https://musicbrainz.org/ws/2/recording/{mbid}?inc=url-rels&fmt=json'
req = urllib.request.Request(url, headers={'User-Agent': 'MIR-Testing/1.0'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(json.dumps(data.get('relations', []), indent=2))
except Exception as e:
    print(e)
