"""Entry point for the web crawler."""

import atexit
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
from argparse import ArgumentParser
from configparser import ConfigParser

from utils import get_logger
from utils.server_registration import get_cache_server
from utils.config import Config
from crawler import Crawler


def main(config_file, restart):
    pidfile = os.path.join("Logs", "crawler.pid")
    os.makedirs("Logs", exist_ok=True)

    # --- restart cleanup (BEFORE logger init so FileHandlers start fresh) ---
    if restart:
        for f in ("crawler_report.json",):
            if os.path.exists(f):
                os.remove(f)
        for name in os.listdir("Logs"):
            if name.endswith(".log") or name == "run.out":
                open(os.path.join("Logs", name), "w").close()

    # --- single-instance guard via PID file --------------------------------
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    try:
        fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(pidfile, "r") as f:
                old_pid = int(f.read().strip())
            if _pid_alive(old_pid):
                print(f"Crawler already running (pid {old_pid}).", file=sys.stderr)
                sys.exit(1)
        except Exception:
            pass
        os.remove(pidfile)
        fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))

    # --- logging / signal setup --------------------------------------------
    logger = get_logger("LAUNCH", "CRAWLER")

    def _cleanup():
        try:
            os.remove(pidfile)
        except OSError:
            pass
        logging.shutdown()

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, _: sys.exit(1))
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda s, _: None)  # ignore

    # --- config & server connection ----------------------------------------
    # macOS defaults to "spawn" which can't pickle spacetime internals
    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    cparser = ConfigParser()
    cparser.read(config_file)
    config = Config(cparser)

    print(f"Connecting to cache server at {config.host}:{config.port} ...")
    result = [None]

    def _connect():
        result[0] = get_cache_server(config, restart)

    t = threading.Thread(target=_connect, daemon=True)
    t.start()
    t.join(timeout=10)
    if t.is_alive() or result[0] is None:
        print("ERROR: Cache server connection timed out.", file=sys.stderr)
        sys.exit(1)
    config.cache_server = result[0]
    print("Connected to cache server.")

    # --- run ---------------------------------------------------------------
    crawler = Crawler(config, restart)
    logger.info(f"Starting crawler with {config.threads_count} threads.")
    try:
        crawler.start()
        logger.info("Crawler finished normally.")
    except Exception as exc:
        logger.exception(f"Crawler crashed: {exc}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--config_file", type=str, default="config.ini")
    args = parser.parse_args()
    main(args.config_file, args.restart)
