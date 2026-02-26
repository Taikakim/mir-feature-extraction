#!/usr/bin/env python3
"""
Tidal API Authentication Utility
Handles the OAuth device flow setup and session loading/saving.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_tidal_session():
    """
    Returns an authenticated tidalapi Session object, 
    or None if tidalapi is not installed or auth fails.
    """
    try:
        import tidalapi
    except ImportError:
        logger.warning("tidalapi not installed. Run: pip install tidalapi")
        return None

    session = tidalapi.Session()
    
    # Define session file path relative to this script
    current_dir = Path(__file__).resolve().parent
    session_file = current_dir / "tidal_session.json"
    
    # Try loading existing session
    if session_file.exists():
        try:
            session.load_session_from_file(str(session_file))
            if session.check_login():
                logger.debug("Tidal session loaded successfully from file")
                return session
            else:
                logger.info("Tidal session in file expired. Requires re-auth.")
        except Exception as e:
            logger.error(f"Failed to load Tidal session: {e}")
    
    # If we get here, we need to authenticate
    print("\n--- TIDAL AUTHENTICATION REQUIRED ---")
    print("A persistent Tidal connection is needed to search for tracks.")
    print("Please follow the link below to link this application to your Tidal account.")
    
    try:
        session.login_oauth_simple()
        if session.check_login():
            print("Authentication successful! Saving session...")
            session.save_session_to_file(str(session_file))
            return session
        else:
            print("Authentication failed.")
            return None
    except Exception as e:
        print(f"Auth error: {e}")
        return None

if __name__ == "__main__":
    setup_logging = logging.basicConfig(level=logging.DEBUG)
    sess = get_tidal_session()
    if sess:
        print(f"Active Session: {sess.session_id}")
        if sess.user:
            print(f"User ID: {sess.user.id}")
