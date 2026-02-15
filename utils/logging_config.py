"""
Centralized Logging Configuration
Provides a consistent logger factory for all modules.
"""

import logging
import sys
import warnings
from pathlib import Path

# Suppress third-party warnings globally
warnings.filterwarnings("ignore", message="Field.*model_.*protected namespace", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Log file path
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "mlops.log"

# Shared formatter
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_INITIALIZED = False


def _init_root() -> None:
    """Configure the root logger once."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (INFO+)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_FORMATTER)
    root.addHandler(console)

    # File handler (DEBUG+)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    root.addHandler(file_handler)

    # Quiet noisy libraries
    for lib in ("mlflow", "urllib3", "git", "botocore", "matplotlib"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with console + file handlers.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    _init_root()
    return logging.getLogger(name)
