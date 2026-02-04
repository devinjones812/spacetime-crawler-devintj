from configparser import ConfigParser
from argparse import ArgumentParser
import atexit
import logging
import multiprocessing as mp
import os
import signal
import sys

from utils import get_logger
from utils.server_registration import get_cache_server
from utils.config import Config
from crawler import Crawler


def main(config_file, restart):
    pidfile = os.path.join("Logs", "crawler.pid")

    def _ensure_logs_dir():
        if not os.path.exists("Logs"):
            os.makedirs("Logs")

    def _pid_is_running(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _read_pidfile():
        try:
            with open(pidfile, "r", encoding="utf-8") as f:
                contents = f.read().strip()
            return int(contents) if contents else None
        except Exception:
            return None

    def _ensure_single_instance():
        _ensure_logs_dir()
        # Try to create lock atomically
        try:
            fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing_pid = _read_pidfile()
            if existing_pid and _pid_is_running(existing_pid):
                print(
                    f"Crawler already running (pid {existing_pid}). "
                    "Stop it before starting a new run.",
                    file=sys.stderr
                )
                sys.exit(1)
            # Stale lockfile; remove and retry
            try:
                os.remove(pidfile)
            except OSError:
                pass
            fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    def _clear_logs():
        _ensure_logs_dir()
        for filename in os.listdir("Logs"):
            if filename.endswith(".log") or filename == "run.out":
                path = os.path.join("Logs", filename)
                try:
                    with open(path, "w", encoding="utf-8"):
                        pass
                except OSError:
                    pass

    _ensure_single_instance()
    _clear_logs()
    logger = get_logger("LAUNCH", "CRAWLER")

    def _on_exit():
        logger.info("Crawler process exiting.")
        try:
            if os.path.exists(pidfile):
                os.remove(pidfile)
        except OSError:
            pass
        logging.shutdown()

    def _handle_signal(signum, _frame):
        logger.warning(f"Received signal {signum}; exiting.")
        sys.exit(1)

    def _handle_sighup(signum, _frame):
        # SIGHUP often comes from terminal/session close; ignore so crawl can continue
        logger.warning(f"Received signal {signum}; ignoring.")

    def _handle_exception(exc_type, exc, tb):
        logger.exception("Unhandled exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    atexit.register(_on_exit)
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        if sig == signal.SIGHUP:
            signal.signal(sig, _handle_sighup)
        else:
            signal.signal(sig, _handle_signal)
    sys.excepthook = _handle_exception

    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    cparser = ConfigParser()
    cparser.read(config_file)
    config = Config(cparser)
    config.cache_server = get_cache_server(config, restart)
    crawler = Crawler(config, restart)
    logger.info(
        f"Starting crawler with {config.threads_count} threads, "
        f"save file '{config.save_file}'.")
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
