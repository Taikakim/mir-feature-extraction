import musicbrainzngs

musicbrainzngs.set_useragent("MIR-Feature-Extraction", "1.0", "https://github.com/user/mir")

def get_tidal_url(track_name, artist_name):
    # Search for the recording
    query = f'recording:"{track_name}" AND artist:"{artist_name}"'
    result = musicbrainzngs.search_recordings(query=query, limit=5)
    
    if not result.get('recording-list'):
        print("No recordings found.")
        return
        
    for recording in result['recording-list']:
        mbid = recording['id']
        print(f"\nChecking MBID: {mbid}")
        
        # Get relationships for this recording
        try:
            rel_data = musicbrainzngs.get_recording_by_id(mbid, includes=['url-rels'])
            if 'recording' in rel_data and 'url-relation-list' in rel_data['recording']:
                for rel in rel_data['recording']['url-relation-list']:
                    url = rel['target']
                    if 'tidal.com' in url:
                        print(f"FOUND TIDAL URL: {url}")
                        return url
        except Exception as e:
            print(f"Error fetching rels for {mbid}: {e}")

if __name__ == "__main__":
    url = get_tidal_url("Get Lucky", "Daft Punk")
    if not url:
        print("Failed to find Tidal URL")
