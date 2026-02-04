import os
import logging
from hashlib import sha256
from urllib.parse import urlparse, urldefrag

def get_logger(name, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not os.path.exists("Logs"):
        os.makedirs("Logs")
    fh = logging.FileHandler(f"Logs/{filename if filename else name}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
       "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_urlhash(url):
    """
    Generate hash for URL, excluding fragment for uniqueness.
    Per assignment spec: URLs with different fragments are considered the same.
    """
    # First defragment the URL
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    # Hash without fragment - netloc, path, params, query only
    return sha256(
        f"{parsed.netloc.lower()}/{parsed.path}/{parsed.params}/"
        f"{parsed.query}".encode("utf-8")).hexdigest()


def normalize(url):
    """Normalize URL by removing trailing slashes and fragments."""
    # Defragment first
    url, _ = urldefrag(url)
    if url.endswith("/") and len(url) > 1:
        return url.rstrip("/")
    return url
