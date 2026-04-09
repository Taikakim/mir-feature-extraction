import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Query a well known track by artist "Daft Punk" - "Get Lucky" to see relationships
# We can search recording!
url = 'https://musicbrainz.org/ws/2/recording/?query=recording:"Get%20Lucky"%20AND%20artist:"Daft%20Punk"&fmt=json'
req = urllib.request.Request(url, headers={'User-Agent': 'MIR-Testing/1.0'})
try:
    with urllib.request.urlopen(req, context=ctx) as response:
        data = json.loads(response.read().decode())
        if data.get('recordings'):
            mbid = data['recordings'][0]['id']
            print(f"Got MBID: {mbid}")
            
            # Now fetch relations
            url2 = f'https://musicbrainz.org/ws/2/recording/{mbid}?inc=url-rels&fmt=json'
            req2 = urllib.request.Request(url2, headers={'User-Agent': 'MIR-Testing/1.0'})
            with urllib.request.urlopen(req2, context=ctx) as r2:
                rel_data = json.loads(r2.read().decode())
                print(json.dumps(rel_data.get('relations', []), indent=2))
        else:
            print("No recordings found")
except Exception as e:
    print(e)
