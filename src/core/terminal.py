"""
Terminal Utilities for MIR Project

ANSI color codes and colored logging for terminal output.

Usage:
    from core.terminal import Colors, fmt_success, setup_colored_logging
    
    # Format text with colors
    print(fmt_success("Operation complete!"))
    print(fmt_error("Something went wrong"))
    
    # Setup colored logging
    logger = setup_colored_logging(level=logging.INFO)
"""

import logging
import sys


class Colors:
    """ANSI escape codes for colored terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright/Light colors
    LIGHT_RED = '\033[91m'
    LIGHT_GREEN = '\033[92m'
    LIGHT_YELLOW = '\033[93m'
    LIGHT_BLUE = '\033[94m'
    LIGHT_MAGENTA = '\033[95m'
    LIGHT_CYAN = '\033[96m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'

    # Semantic aliases
    HEADER = LIGHT_GREEN + BOLD
    FILENAME = YELLOW
    STAGE = LIGHT_CYAN + BOLD
    SUCCESS = GREEN
    WARNING = YELLOW  # Orange not available, yellow is close
    ERROR = LIGHT_RED
    NOTIFICATION = LIGHT_YELLOW
    PROGRESS = CYAN
    DIM_TEXT = DIM


def color(text: str, color_code: str) -> str:
    """Wrap text with color code and reset."""
    return f"{color_code}{text}{Colors.RESET}"


def fmt_filename(name: str) -> str:
    """Format a filename with yellow color."""
    return color(name, Colors.FILENAME)


def fmt_header(text: str) -> str:
    """Format a header with light green bold."""
    return color(text, Colors.HEADER)


def fmt_stage(text: str) -> str:
    """Format a stage name with cyan bold."""
    return color(text, Colors.STAGE)


def fmt_success(text: str) -> str:
    """Format success message with green."""
    return color(text, Colors.SUCCESS)


def fmt_warning(text: str) -> str:
    """Format warning with yellow/orange."""
    return color(text, Colors.WARNING)


def fmt_error(text: str) -> str:
    """Format error with light red."""
    return color(text, Colors.ERROR)


def fmt_progress(text: str) -> str:
    """Format progress info with cyan."""
    return color(text, Colors.PROGRESS)


def fmt_notification(text: str) -> str:
    """Format notification with light yellow."""
    return color(text, Colors.NOTIFICATION)


def fmt_dim(text: str) -> str:
    """Format text as dimmed."""
    return color(text, Colors.DIM_TEXT)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level and content."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM,
        logging.INFO: Colors.RESET,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.LIGHT_RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE,
    }

    def format(self, record):
        # Get base color for log level
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)

        # Format the message
        msg = record.getMessage()

        # Apply special formatting based on content
        if record.levelno >= logging.ERROR:
            msg = f"{Colors.LIGHT_RED}{msg}{Colors.RESET}"
        elif record.levelno >= logging.WARNING:
            msg = f"{Colors.YELLOW}{msg}{Colors.RESET}"
        elif '=====' in msg or '-----' in msg:
            # Headers/separators
            msg = f"{Colors.LIGHT_GREEN}{Colors.BOLD}{msg}{Colors.RESET}"
        elif msg.startswith('[STAGE') or msg.startswith('[1') or msg.startswith('[2'):
            # Stage markers
            msg = f"{Colors.LIGHT_CYAN}{Colors.BOLD}{msg}{Colors.RESET}"
        elif 'Progress:' in msg or '%' in msg:
            # Progress updates
            msg = f"{Colors.CYAN}{msg}{Colors.RESET}"
        elif 'complete' in msg.lower() or 'success' in msg.lower() or 'finished' in msg.lower():
            # Success messages
            msg = f"{Colors.GREEN}{msg}{Colors.RESET}"
        elif 'skip' in msg.lower():
            # Skipped items
            msg = f"{Colors.DIM}{msg}{Colors.RESET}"

        # Format timestamp and level
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname

        if record.levelno >= logging.WARNING:
            level = f"{level_color}{level}{Colors.RESET}"

        return f"{Colors.DIM}{timestamp}{Colors.RESET} {level}: {msg}"


def setup_colored_logging(level=logging.INFO):
    """Setup logging with colored output for the master pipeline."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(datefmt='%H:%M:%S'))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers and add colored one
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    # Also configure module logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    return logger
