"""
Graceful shutdown via keypress for long-running pipelines.

Usage:
    from core.graceful_shutdown import shutdown_requested, start_shutdown_listener

    start_shutdown_listener()  # call once at pipeline start

    for item in work:
        if shutdown_requested.is_set():
            logger.info("Shutdown requested, finishing current pass...")
            break
        process(item)
"""

import atexit
import logging
import os
import select
import sys
import threading

logger = logging.getLogger(__name__)

# Global event — check this in processing loops
shutdown_requested = threading.Event()

_listener_thread = None
_old_termios = None


def _restore_terminal():
    """Restore terminal settings on exit."""
    global _old_termios
    if _old_termios is not None:
        try:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_termios)
            _old_termios = None
        except Exception:
            pass


def _listener_loop(trigger_key: str):
    """Background thread: watch stdin for the trigger key."""
    global _old_termios
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        _old_termios = termios.tcgetattr(fd)
        # cbreak mode: single keypress, no echo, but Ctrl-C still works
        tty.setcbreak(fd)

        while not shutdown_requested.is_set():
            # Wait up to 0.5s for input so we can check the event periodically
            ready, _, _ = select.select([fd], [], [], 0.5)
            if ready:
                ch = os.read(fd, 1).decode('utf-8', errors='ignore')
                if ch.lower() == trigger_key.lower():
                    shutdown_requested.set()
                    logger.info(f"\n>>> Graceful shutdown requested ('{trigger_key}' pressed). "
                                f"Finishing current file...")
                    break
    except Exception:
        # Not a terminal (piped input, etc.) — fall back silently
        pass
    finally:
        _restore_terminal()


def start_shutdown_listener(trigger_key: str = 's'):
    """
    Start listening for a keypress to trigger graceful shutdown.

    Args:
        trigger_key: The key that triggers shutdown (default: 's')
    """
    global _listener_thread

    if _listener_thread is not None and _listener_thread.is_alive():
        return  # already running

    # Only works with a real terminal
    if not sys.stdin.isatty():
        return

    shutdown_requested.clear()
    atexit.register(_restore_terminal)

    _listener_thread = threading.Thread(
        target=_listener_loop,
        args=(trigger_key,),
        daemon=True,
        name='shutdown-listener',
    )
    _listener_thread.start()
    logger.info(f"Press '{trigger_key}' at any time to stop after the current file finishes.")


def stop_shutdown_listener():
    """Stop the listener and restore terminal."""
    global _listener_thread
    shutdown_requested.set()  # signal thread to exit
    _restore_terminal()
    if _listener_thread is not None:
        _listener_thread.join(timeout=1.0)
        _listener_thread = None
    shutdown_requested.clear()  # reset for potential reuse
