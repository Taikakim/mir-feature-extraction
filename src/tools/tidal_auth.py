#!/usr/bin/env python3
"""
Tidal API Authentication Utility
Handles the OAuth device flow setup and session loading/saving.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level singleton — avoids re-authenticating on every lookup_track() call.
# _TIDAL_UNAVAILABLE is a sentinel meaning "already tried, don't try again this run".
_tidal_session_cache = None
_TIDAL_UNAVAILABLE = object()


def get_tidal_session():
    """
    Returns an authenticated tidalapi Session object,
    or None if tidalapi is not installed or auth fails.

    Uses a module-level singleton so auth happens at most ONCE per process.
    After the first attempt (success or failure) all subsequent calls return
    immediately without any I/O, network checks, or prompts.
    """
    global _tidal_session_cache

    # Already resolved this run — return immediately (None if previously failed)
    if _tidal_session_cache is _TIDAL_UNAVAILABLE:
        return None
    if _tidal_session_cache is not None:
        return _tidal_session_cache

    try:
        import tidalapi
    except ImportError:
        logger.warning("tidalapi not installed. Run: pip install tidalapi")
        _tidal_session_cache = _TIDAL_UNAVAILABLE
        return None

    session = tidalapi.Session()
    session_file = Path(__file__).resolve().parent / "tidal_session.json"

    # Try loading existing session from disk
    if session_file.exists():
        try:
            session.load_session_from_file(str(session_file))
            if session.check_login():
                logger.debug("Tidal session loaded from file")
                _tidal_session_cache = session
                return session
            else:
                logger.info("Tidal session expired — re-authentication required")
        except Exception as e:
            logger.error(f"Failed to load Tidal session: {e}")

    # Interactive OAuth device flow (runs at most once per process)
    print("\n--- TIDAL AUTHENTICATION REQUIRED ---")
    print("A persistent Tidal connection is needed to search for tracks.")
    print("Please follow the link below to link this application to your Tidal account.")

    try:
        session.login_oauth_simple()
        if session.check_login():
            print("Authentication successful! Saving session...")
            session.save_session_to_file(str(session_file))
            _tidal_session_cache = session
            return session
        else:
            print("Authentication failed.")
    except Exception as e:
        print(f"Auth error: {e}")

    # Mark as unavailable so we never prompt again this run
    _tidal_session_cache = _TIDAL_UNAVAILABLE
    return None

if __name__ == "__main__":
    setup_logging = logging.basicConfig(level=logging.DEBUG)
    sess = get_tidal_session()
    if sess:
        print(f"Active Session: {sess.session_id}")
        if sess.user:
            print(f"User ID: {sess.user.id}")
