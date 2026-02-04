"""
Thread-safe Frontier with Per-Domain Politeness.

This module implements a thread-safe URL frontier that manages:
- URL queue with thread-safe operations
- Per-domain politeness (500ms delay between requests to same domain)
- Deduplication of URLs
- Persistence via shelve

Extra Credit: Multithreading support (+5 points)
"""

import os
import shelve
import time
from threading import RLock, Condition
from collections import defaultdict
from queue import Queue, Empty
from urllib.parse import urlparse

from utils import get_logger, get_urlhash, normalize
from scraper import is_valid


class Frontier(object):
    """
    Thread-safe Frontier implementation with per-domain politeness.
    
    Key features:
    - Thread-safe URL queue operations
    - Per-domain politeness delay tracking
    - Multiple domain queues for efficient multithreaded crawling
    - Persistence for crash recovery
    """
    
    def __init__(self, config, restart):
        self.logger = get_logger("FRONTIER")
        self.config = config
        
        # Thread safety locks
        self._lock = RLock()  # Main lock for shared state
        self._condition = Condition(self._lock)  # For waiting on new URLs
        
        # Per-domain queues and timing
        self._domain_queues = defaultdict(list)  # domain -> list of URLs
        self._domain_last_access = defaultdict(float)  # domain -> last access time
        self._domain_locks = defaultdict(RLock)  # domain -> lock
        
        # All URLs for deduplication
        self._seen_urls = set()
        
        # Track in-flight URLs (checked out but not yet completed)
        self._in_progress = 0
        self._finished = False
        
        # Politeness delay in seconds
        self._politeness_delay = config.time_delay  # Default 0.5s
        
        # Setup persistence
        if not os.path.exists(self.config.save_file) and not restart:
            self.logger.info(
                f"Did not find save file {self.config.save_file}, "
                f"starting from seed.")
        elif os.path.exists(self.config.save_file) and restart:
            self.logger.info(
                f"Found save file {self.config.save_file}, deleting it.")
            os.remove(self.config.save_file)
        
        # Load existing save file, or create one if it does not exist
        self.save = shelve.open(self.config.save_file)
        
        if restart:
            for url in self.config.seed_urls:
                self.add_url(url)
        else:
            self._parse_save_file()
            if not self.save:
                for url in self.config.seed_urls:
                    self.add_url(url)
    
    def _get_domain(self, url):
        """Extract domain from URL for politeness tracking."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return "unknown"
    
    def _parse_save_file(self):
        """Load state from save file."""
        total_count = len(self.save)
        tbd_count = 0
        
        for url, completed in self.save.values():
            self._seen_urls.add(get_urlhash(url))
            if not completed and is_valid(url):
                domain = self._get_domain(url)
                self._domain_queues[domain].append(url)
                tbd_count += 1
        
        self.logger.info(
            f"Found {tbd_count} urls to be downloaded from {total_count} "
            f"total urls discovered.")
    
    def _get_total_pending(self):
        """Get total number of pending URLs across all domains."""
        return sum(len(q) for q in self._domain_queues.values())
    
    def get_tbd_url(self):
        """
        Get one URL that needs to be downloaded, respecting per-domain politeness.
        
        Returns None if:
        - All URLs are exhausted and all workers are done
        - Or if we should stop crawling
        
        This method may block if all domains are currently rate-limited.
        """
        while True:
            with self._lock:
                # Check if we should stop
                if self._finished:
                    return None

                # Find a domain that's ready (respects politeness)
                current_time = time.time()
                best_url = None
                best_domain = None
                min_wait = float('inf')

                for domain, urls in list(self._domain_queues.items()):
                    if not urls:
                        continue

                    last_access = self._domain_last_access.get(domain, 0)
                    time_since = current_time - last_access

                    if time_since >= self._politeness_delay:
                        # This domain is ready
                        best_url = urls.pop(0)
                        best_domain = domain
                        break
                    else:
                        # Track minimum wait time
                        wait_time = self._politeness_delay - time_since
                        if wait_time < min_wait:
                            min_wait = wait_time

                if best_url:
                    # Update last access time and mark in-flight
                    self._domain_last_access[best_domain] = time.time()
                    self._in_progress += 1
                    return best_url

                # Check if there are any URLs left
                total_pending = self._get_total_pending()

                if total_pending == 0:
                    # No pending URLs; wait for in-progress workers to add more
                    if self._in_progress == 0:
                        stats = self.get_stats()
                        self.logger.info(
                            "Frontier empty; stopping crawl. "
                            f"Seen={stats['total_seen']}, "
                            f"Pending={stats['pending']}, "
                            f"InProgress={stats['in_progress']}, "
                            f"Domains={stats['domains']}")
                        self._finished = True
                        self._condition.notify_all()
                        return None
                    self._condition.wait(timeout=1.0)
                    continue

                # URLs exist but all domains are rate-limited
                # Wait for the minimum time needed
                if min_wait < float('inf'):
                    wait_time = min(min_wait + 0.01, 0.5)
                    self._condition.wait(timeout=wait_time)
                else:
                    self._condition.wait(timeout=0.1)
    
    def add_url(self, url):
        """
        Add a URL to the frontier if not already seen.
        Thread-safe with notification for waiting workers.
        """
        url = normalize(url)
        urlhash = get_urlhash(url)
        
        with self._lock:
            if urlhash in self._seen_urls:
                return  # Already seen
            
            # Mark as seen
            self._seen_urls.add(urlhash)
            
            # Save to persistence
            self.save[urlhash] = (url, False)
            self.save.sync()
            
            # Add to domain queue
            domain = self._get_domain(url)
            self._domain_queues[domain].append(url)
            
            # Notify waiting workers
            self._condition.notify_all()
    
    def mark_url_complete(self, url):
        """Mark a URL as completed."""
        urlhash = get_urlhash(url)
        with self._lock:
            if urlhash not in self.save:
                self.logger.error(
                    f"Completed url {url}, but have not seen it before.")
            
            self.save[urlhash] = (url, True)
            self.save.sync()
            if self._in_progress > 0:
                self._in_progress -= 1
            self._condition.notify_all()
    
    def get_stats(self):
        """Get frontier statistics."""
        with self._lock:
            return {
                "total_seen": len(self._seen_urls),
                "pending": self._get_total_pending(),
                "domains": len(self._domain_queues),
                "in_progress": self._in_progress
            }
