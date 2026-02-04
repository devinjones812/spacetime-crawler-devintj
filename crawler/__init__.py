import os
import time
from threading import Event, Thread

from utils import get_logger
from crawler.frontier import Frontier
from crawler.worker import Worker

class Crawler(object):
    def __init__(self, config, restart, frontier_factory=Frontier, worker_factory=Worker):
        self.config = config
        self.logger = get_logger("CRAWLER")
        self.frontier = frontier_factory(config, restart)
        self.workers = list()
        self.worker_factory = worker_factory

    def start_async(self):
        self.workers = [
            self.worker_factory(worker_id, self.config, self.frontier)
            for worker_id in range(self.config.threads_count)]
        for worker in self.workers:
            worker.start()

    def start(self):
        stop_event = Event()
        heartbeat = Thread(
            target=self._heartbeat_loop,
            args=(stop_event,),
            daemon=True
        )
        heartbeat.start()
        self.start_async()
        self.join()
        stop_event.set()
        self.logger.info("All workers joined; crawler stopping.")

    def join(self):
        for worker in self.workers:
            worker.join()

    def _heartbeat_loop(self, stop_event, interval=10.0):
        """Periodically log frontier stats to detect stalls."""
        pid = os.getpid()
        while not stop_event.is_set():
            stats = self.frontier.get_stats()
            self.logger.info(
                "Heartbeat(pid=%s): pending=%s in_progress=%s seen=%s domains=%s",
                pid,
                stats["pending"],
                stats["in_progress"],
                stats["total_seen"],
                stats["domains"],
            )
            stop_event.wait(interval)
