# Spacetime Crawler

A multithreaded web crawler built on the spacetime cache server framework. Crawls `*.ics.uci.edu`, `*.cs.uci.edu`, `*.informatics.uci.edu`, and `*.stat.uci.edu`, collecting analytics and producing a structured report.

## Features

- **Multithreaded crawling** (4 workers) with per-domain politeness (500ms delay)
- **Thread-safe frontier** with per-domain queues — workers never block each other waiting on politeness for the same domain
- **Trap detection** — regex-based pattern matching (calendars, pagination, GitLab internals, etc.) plus URL pattern frequency limits
- **Duplicate detection** — exact (SHA-256 content hashing) and near-duplicate (SimHash with bit-sampling index, implemented from scratch)
- **Crash recovery** — crawler state is saved periodically and restored on restart
- **Soft-404 detection** and low-information page filtering

## Setup

Requires Python 3.6+.

Install dependencies:

```
python -m pip install packages/spacetime-2.1.1-py3-none-any.whl
python -m pip install -r packages/requirements.txt
```

## Configuration

All settings live in `config.ini`:

| Setting       | Description                                      |
|---------------|--------------------------------------------------|
| `USERAGENT`   | Identifies the crawler to the cache server       |
| `HOST`/`PORT` | Cache server address                             |
| `SEEDURL`     | Comma-separated starting URLs                    |
| `POLITENESS`  | Per-domain delay between requests (seconds)      |
| `SAVE`        | Base filename for frontier persistence            |
| `THREADCOUNT` | Number of worker threads (max 4)                 |

## Usage

Start the crawler:

```
python launch.py
```

Restart from scratch (deletes all progress):

```
python launch.py --restart
```

Use a different config file:

```
python launch.py --config_file path/to/config.ini
```

## Output Files

| File                   | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| `crawler_report.json`  | Clean final report — unique page count, longest page, top 50 words, subdomains |
| `crawler_state.json`   | Full crawler state for resume — includes all word frequencies      |
| `frontier.shelve.pkl`  | Frontier persistence — tracks which URLs have been seen/completed  |
| `Logs/Worker.log`      | Per-worker download and processing logs                            |
| `Logs/FRONTIER.log`    | Frontier lifecycle events                                          |
| `Logs/CRAWLER.log`     | Top-level crawler and heartbeat logs                               |

Both `crawler_report.json` and `crawler_state.json` are saved every 50 unique pages during the crawl, so progress is preserved even if the server goes down.

## Architecture

```
launch.py
  └── Crawler  (crawler/__init__.py)
        ├── Frontier  (crawler/frontier.py)
        │     ├── Per-domain URL queues
        │     ├── Politeness tracking
        │     └── Pickle-backed persistence
        └── Worker × 4  (crawler/worker.py)
              ├── Downloads via cache server
              └── Calls scraper() for each page

scraper.py
  ├── CrawlerAnalytics  — thread-safe stats tracking
  ├── SimHash index     — near-duplicate detection (from scratch)
  ├── Trap detection    — regex patterns + frequency limits
  ├── is_valid()        — URL filtering (domain, extension, traps)
  └── scraper()         — parse page, record analytics, extract links
```

**Flow:** Workers pull URLs from the frontier (which picks a domain whose politeness window has elapsed), download via the cache server, pass the response to `scraper()`, and feed discovered URLs back to the frontier. The cycle ends when no URLs remain pending or in-flight.

## Extra Credit

- **SimHash near-duplicate detection** (+2 pts) — 64-bit SimHash with a 4-band bit-sampling index for O(1) average-case lookups. Implemented from scratch with no external libraries.
- **Multithreaded crawling** (+5 pts) — 4 worker threads with a thread-safe frontier that enforces 500ms per-domain politeness. Uses `RLock`/`Condition` for synchronization and per-domain queues so threads can work on different domains in parallel.
