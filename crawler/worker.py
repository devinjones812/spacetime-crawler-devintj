"""
Worker thread for multithreaded crawling.

Downloads URLs via the cache server, passes responses to the scraper,
and feeds discovered URLs back to the frontier.  Politeness is enforced
by the Frontier (per-domain delay tracking).

Extra credit: multithreading support (+5 pts).
"""

from inspect import getsource
from threading import Thread

from utils import get_logger
from utils.download import download
import scraper


class Worker(Thread):
    def __init__(self, worker_id, config, frontier):
        self.worker_id = worker_id
        self.logger = get_logger(f"Worker-{worker_id}", "Worker")
        self.config = config
        self.frontier = frontier

        # Verify scraper doesn't make its own HTTP requests
        src = getsource(scraper)
        for banned in ("from requests import", "import requests",
                       "from urllib.request import", "import urllib.request"):
            assert banned not in src, f"Do not use {banned.split()[1]} in scraper.py"

        super().__init__(daemon=True)

    def run(self):
        processed = 0
        while True:
            tbd_url = self.frontier.get_tbd_url()
            if tbd_url is None:
                self.logger.info(f"Frontier empty after {processed} URLs. Exiting.")
                return

            try:
                resp = download(tbd_url, self.config, self.logger)

                if resp.status == 599:
                    self.logger.warning(f"Server error for {tbd_url}, re-queuing.")
                    self.frontier.mark_url_failed(tbd_url)
                    continue

                self.logger.info(
                    f"Downloaded {tbd_url}, status <{resp.status}>.")

                scraped_urls = scraper.scraper(tbd_url, resp)
                self.frontier.complete_work_cycle(tbd_url, scraped_urls)
                processed += 1

                if processed % 50 == 0:
                    stats = self.frontier.get_stats()
                    self.logger.info(
                        f"Worker-{self.worker_id}: {processed} done, "
                        f"{stats['pending']} pending, {stats['total_seen']} seen")

            except Exception as e:
                self.logger.error(f"Error processing {tbd_url}: {e}")
                self.frontier.mark_url_failed(tbd_url)
