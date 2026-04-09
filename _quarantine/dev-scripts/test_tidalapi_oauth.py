import tidalapi
import time
import os

def authenticate():
    session = tidalapi.Session()
    # The default behavior of login_oauth_simple is to generate a device link
    print("Initiating OAuth login process...")
    session.login_oauth_simple()
    
    if session.check_login():
        print("Login successful!")
        
        # Test searching by ISRC
        try:
            tracks = session.search('track', 'USQX91300108')
            if hasattr(tracks, 'get') and 'tracks' in tracks:
                print("Found track length:", len(tracks['tracks']))
                if tracks['tracks']:
                    print("Track Name:", tracks['tracks'][0].name)
        except Exception as e:
            print("Search failed:", e)
            
        try:
            tracks2 = session.get_tracks_by_isrc("USQX91300108")
            if tracks2:
                print("get_tracks_by_isrc Success:", tracks2[0].name)
        except Exception as e:
            print("get_tracks_by_isrc failed:", e)
            
        print("\nSession Details:")
        print("Access Token:", session.access_token[:10] + '...')
        print("Refresh Token:", session.refresh_token[:10] + '...')
        print("Session ID:", session.session_id)
        
        # We can save this using the existing save_session_to_file if required
        # session.save_session_to_file("tidal_session.json")
    else:
        print("Login failed.")

if __name__ == "__main__":
    authenticate()
