import os
from threading import Event, Thread

from utils import get_logger
from crawler.frontier import Frontier
from crawler.worker import Worker


class Crawler:
    def __init__(self, config, restart, frontier_factory=Frontier, worker_factory=Worker):
        self.config = config
        self.logger = get_logger("CRAWLER")
        self.frontier = frontier_factory(config, restart)
        self.worker_factory = worker_factory

    def start(self):
        stop = Event()
        Thread(target=self._heartbeat, args=(stop,), daemon=True).start()

        workers = [
            self.worker_factory(i, self.config, self.frontier)
            for i in range(self.config.threads_count)
        ]
        for w in workers:
            w.start()
        for i, w in enumerate(workers):
            w.join()
            self.logger.info(f"Worker-{i} joined.")

        self.frontier.save.sync()
        stop.set()
        self.logger.info("All workers joined; crawler stopping.")

    def _heartbeat(self, stop, interval=10.0):
        while not stop.is_set():
            s = self.frontier.get_stats()
            self.logger.info(
                "Heartbeat(pid=%s): pending=%s in_progress=%s seen=%s "
                "active_domains=%s total_domains=%s",
                os.getpid(), s["pending"], s["in_progress"],
                s["total_seen"], s["domains_with_pending"],
                s["domains_ever_seen"],
            )
            stop.wait(interval)
