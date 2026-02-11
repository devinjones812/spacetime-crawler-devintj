"""
Thread-safe Frontier with per-domain politeness.

Manages the URL queue, deduplication, per-domain rate limiting (500 ms),
and persistence via pickle.  Extra credit: multithreading support (+5 pts).
"""

import os
import time
import pickle
from collections import defaultdict, deque
from threading import RLock, Condition
from urllib.parse import urlparse

from utils import get_logger, get_urlhash, normalize


# ---------------------------------------------------------------------------
# Persistent dictionary (pickle-backed, thread-safe)
# ---------------------------------------------------------------------------

class _PersistentDict:
    """Thread-safe dict that periodically syncs to a pickle file."""

    _SYNC_INTERVAL = 10  # seconds between automatic syncs

    def __init__(self, filename):
        self.filename = filename
        self.lock = RLock()
        self.data = {}
        self._last_sync = time.time()
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    self.data = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                self.data = {}

    def __setitem__(self, key, value):
        with self.lock:
            self.data[key] = value

    def __getitem__(self, key):
        with self.lock:
            return self.data[key]

    def __contains__(self, key):
        with self.lock:
            return key in self.data

    def __len__(self):
        with self.lock:
            return len(self.data)

    def __bool__(self):
        with self.lock:
            return bool(self.data)

    def values(self):
        with self.lock:
            return list(self.data.values())

    def sync(self):
        """Persist to disk unconditionally."""
        with self.lock:
            snapshot = dict(self.data)
        with open(self.filename, "wb") as f:
            pickle.dump(snapshot, f)
        with self.lock:
            self._last_sync = time.time()

    def maybe_sync(self):
        """Persist only if enough time has elapsed since the last sync."""
        with self.lock:
            if time.time() - self._last_sync < self._SYNC_INTERVAL:
                return
            snapshot = dict(self.data)
            self._last_sync = time.time()
        with open(self.filename, "wb") as f:
            pickle.dump(snapshot, f)


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------

class Frontier:
    """
    Thread-safe URL frontier with per-domain politeness.

    Workers call get_tbd_url() (blocks until a URL is ready) and
    complete_work_cycle() (atomically adds discovered URLs and marks
    the source URL done).
    """

    def __init__(self, config, restart):
        self.logger = get_logger("FRONTIER")
        self.config = config

        self._lock = RLock()
        self._cond = Condition(self._lock)

        self._domain_queues = defaultdict(deque)   # domain -> deque of urls
        self._domain_last_access = defaultdict(float)
        self._seen = set()
        self._in_progress = 0
        self._shutdown = False
        self._politeness = config.time_delay       # 0.5 s

        # Persistence
        save_path = config.save_file + ".pkl"
        if os.path.exists(save_path) and restart:
            self.logger.info(f"Deleting old save file {save_path}.")
            os.remove(save_path)
        self.save = _PersistentDict(save_path)

        if restart or not self.save:
            for url in config.seed_urls:
                self.add_url(url)
        else:
            self._restore_from_save()

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _domain_of(url):
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return "unknown"

    def _restore_from_save(self):
        tbd = 0
        for url, completed in self.save.values():
            self._seen.add(get_urlhash(url))
            if not completed:
                self._domain_queues[self._domain_of(url)].append(url)
                tbd += 1
        self.logger.info(f"Restored {tbd} pending URLs from {len(self.save)} total.")

    def _total_pending(self):
        return sum(len(q) for q in self._domain_queues.values())

    def _enqueue(self, url):
        """Add url if unseen.  Must be called while holding _lock."""
        url = normalize(url)
        h = get_urlhash(url)
        if h in self._seen:
            return
        self._seen.add(h)
        self.save[h] = (url, False)
        self.save.maybe_sync()
        self._domain_queues[self._domain_of(url)].append(url)

    # -- public API ---------------------------------------------------------

    def add_url(self, url):
        with self._lock:
            self._enqueue(url)
            self._cond.notify_all()

    def get_tbd_url(self):
        """
        Return the next URL to crawl, blocking until one is available.
        Returns None when the crawl is truly finished.
        """
        while True:
            with self._lock:
                # If another thread already declared shutdown, exit immediately
                if self._shutdown:
                    return None

                now = time.time()
                min_wait = float("inf")

                for domain, urls in self._domain_queues.items():
                    if not urls:
                        continue
                    elapsed = now - self._domain_last_access.get(domain, 0)
                    if elapsed >= self._politeness:
                        best_url = urls.popleft()
                        self._domain_last_access[domain] = time.time()
                        self._in_progress += 1
                        return best_url
                    wait = self._politeness - elapsed
                    if wait < min_wait:
                        min_wait = wait

                pending = self._total_pending()

                # Truly done: nothing pending and nothing in-flight
                if pending == 0 and self._in_progress == 0:
                    self._cond.wait(timeout=2.0)
                    if self._total_pending() == 0 and self._in_progress == 0:
                        self._shutdown = True
                        self._cond.notify_all()  # wake ALL blocked workers
                        self.logger.info("Frontier empty — crawl complete.")
                        return None
                    continue

                # Wait for politeness or new work
                if pending == 0:
                    self._cond.wait(timeout=1.0)
                elif min_wait < float("inf"):
                    self._cond.wait(timeout=min(min_wait + 0.01, 0.5))
                else:
                    self._cond.wait(timeout=0.1)

    def complete_work_cycle(self, url, discovered_urls):
        """Atomically add discovered URLs and mark *url* as completed."""
        with self._lock:
            for new_url in discovered_urls:
                self._enqueue(new_url)

            h = get_urlhash(normalize(url))
            if h in self.save:
                self.save[h] = (normalize(url), True)
                self.save.maybe_sync()

            if self._in_progress > 0:
                self._in_progress -= 1
            self._cond.notify_all()

    def mark_url_failed(self, url):
        """Re-queue a URL that failed (e.g. server error)."""
        with self._lock:
            self._domain_queues[self._domain_of(url)].append(url)
            if self._in_progress > 0:
                self._in_progress -= 1
            self._cond.notify_all()

    def get_stats(self):
        with self._lock:
            return {
                "total_seen": len(self._seen),
                "pending": self._total_pending(),
                "in_progress": self._in_progress,
                "domains_with_pending": sum(1 for q in self._domain_queues.values() if q),
                "domains_ever_seen": len(self._domain_queues),
            }
