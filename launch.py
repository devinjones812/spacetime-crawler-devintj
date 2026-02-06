from configparser import ConfigParser
from argparse import ArgumentParser
import atexit
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading

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

    # Attempt to connect to cache server with timeout
    print(f"Connecting to cache server at {config.host}:{config.port}...")
    print(f"Using user agent: {config.user_agent}")
    print("Waiting for server response (timeout: 10 seconds)...")

    cache_server = [None]
    error = [None]

    def get_server():
        try:
            cache_server[0] = get_cache_server(config, restart)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=get_server, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        # Server didn't respond in time
        print("\n" + "="*80)
        print("ERROR: Cache server connection timeout after 10 seconds.")
        print(f"\nThe UCI cache server ({config.host}:{config.port}) is likely DOWN.")
        print("\nWhat to do:")
        print("  1. Check Ed Discussion for server status updates")
        print("  2. Ask your TA/instructor if the server is running")
        print("  3. Verify your user agent in config.ini is correct")
        print("  4. Try again later when the server is back online")
        print("="*80 + "\n")
        sys.exit(1)

    if error[0]:
        raise error[0]

    config.cache_server = cache_server[0]
    print("✓ Successfully connected to cache server!")

    if restart:
        import scraper
        scraper.analytics.reset()
        with scraper.exact_hashes_lock:
            scraper.exact_hashes.clear()
        for f in ("crawler_report.json", "trap_report.json"):
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed stale {f}")

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
