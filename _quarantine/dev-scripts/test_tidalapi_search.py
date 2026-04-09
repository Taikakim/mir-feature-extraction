import tidalapi
import os

def test():
    session = tidalapi.Session()
    # Tidal API does not allow search without login anymore, but let's try with an API token or device flow...
    # The client credentials flow isn't natively supported, but maybe search doesn't strictly need auth for open endpoints?
    try:
        print("Trying search...")
        # tidalapi requires the model class for type
        tracks = session.search('Daft Punk Get Lucky', models=[tidalapi.models.Track])
        print("Search success:", tracks)
    except Exception as e:
        print("Search failed:", e)

    print("\nSession attributes:", [a for a in dir(session) if not a.startswith('_')])

if __name__ == "__main__":
    test()
