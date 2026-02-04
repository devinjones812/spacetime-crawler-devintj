"""
Thread-safe Worker for Multithreaded Crawling.

This module implements a worker thread that:
- Downloads URLs from the cache server
- Passes responses to the scraper
- Adds discovered URLs back to the frontier
- Respects politeness delays (handled by frontier)

Extra Credit: Multithreading support (+5 points)
"""

from threading import Thread

from inspect import getsource
from utils.download import download
from utils import get_logger
import scraper
import time


class Worker(Thread):
    """
    Worker thread for downloading and processing URLs.
    
    Politeness is enforced by the Frontier, which tracks per-domain
    access times and only returns URLs when it's safe to crawl them.
    """
    
    def __init__(self, worker_id, config, frontier):
        self.worker_id = worker_id
        self.logger = get_logger(f"Worker-{worker_id}", "Worker")
        self.config = config
        self.frontier = frontier
        
        # Basic check for requests in scraper (prevents direct web requests)
        assert {getsource(scraper).find(req) for req in {"from requests import", "import requests"}} == {-1}, \
            "Do not use requests in scraper.py"
        assert {getsource(scraper).find(req) for req in {"from urllib.request import", "import urllib.request"}} == {-1}, \
            "Do not use urllib.request in scraper.py"
        
        super().__init__(daemon=True)
    
    def run(self):
        """Main worker loop - download URLs and process them."""
        urls_processed = 0
        
        while True:
            # Get next URL from frontier (may block for politeness)
            tbd_url = self.frontier.get_tbd_url()
            
            if not tbd_url:
                self.logger.info(
                    f"Frontier is empty. Worker-{self.worker_id} stopping. "
                    f"Processed {urls_processed} URLs.")
                break
            
            try:
                # Download the URL
                resp = download(tbd_url, self.config, self.logger)
                
                self.logger.info(
                    f"Downloaded {tbd_url}, status <{resp.status}>, "
                    f"using cache {self.config.cache_server}.")
                
                # Process with scraper
                scraped_urls = scraper.scraper(tbd_url, resp)
                
                # Add discovered URLs to frontier
                for scraped_url in scraped_urls:
                    self.frontier.add_url(scraped_url)
                
                # Mark as complete
                self.frontier.mark_url_complete(tbd_url)
                
                urls_processed += 1
                
                # Log progress periodically
                if urls_processed % 50 == 0:
                    stats = self.frontier.get_stats()
                    self.logger.info(
                        f"Worker-{self.worker_id} progress: {urls_processed} processed, "
                        f"Frontier: {stats['pending']} pending, {stats['total_seen']} seen")
            
            except Exception as e:
                self.logger.error(f"Error processing {tbd_url}: {e}")
                self.frontier.mark_url_complete(tbd_url)
            
            # Note: Politeness delay is now handled by the Frontier's per-domain tracking
            # The frontier will not return a URL for a domain until the delay has passed
            # This allows workers to immediately process URLs from different domains
