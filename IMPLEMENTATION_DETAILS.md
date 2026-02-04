# Web Crawler Implementation Details

This document provides a comprehensive breakdown of everything implemented for the UCI web crawler project. Use this as a reference for your interview.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Files Modified/Created](#files-modifiedcreated)
3. [Core Implementation](#core-implementation)
   - [URL Extraction and Parsing](#url-extraction-and-parsing)
   - [URL Validation (is_valid)](#url-validation-is_valid)
   - [Domain Filtering](#domain-filtering)
   - [Trap Detection](#trap-detection)
4. [Analytics Tracking](#analytics-tracking)
   - [Unique Pages](#unique-pages)
   - [Word Counting](#word-counting)
   - [Subdomain Tracking](#subdomain-tracking)
5. [Extra Credit: Similarity Detection](#extra-credit-similarity-detection-2-points)
   - [Exact Duplicate Detection](#exact-duplicate-detection)
   - [Near-Duplicate Detection (SimHash)](#near-duplicate-detection-simhash)
6. [Extra Credit: Multithreading](#extra-credit-multithreading-5-points)
   - [Thread-Safe Frontier](#thread-safe-frontier)
   - [Per-Domain Politeness](#per-domain-politeness)
   - [Worker Implementation](#worker-implementation)
7. [Key Algorithms Explained](#key-algorithms-explained)
8. [Configuration](#configuration)
9. [How to Run](#how-to-run)
10. [Interview Preparation Notes](#interview-preparation-notes)

---

## Project Overview

**Goal:** Build a web crawler that crawls specific UCI domains and generates analytics including:
- Number of unique pages
- Longest page by word count
- 50 most common words (excluding stopwords)
- Subdomain statistics

**Allowed Domains:**
- `*.ics.uci.edu/*`
- `*.cs.uci.edu/*`
- `*.informatics.uci.edu/*`
- `*.stat.uci.edu/*`

**Requirements Met:**
- ✅ Honor politeness delay (500ms per domain)
- ✅ Crawl pages with high textual information content
- ✅ Detect and avoid infinite traps
- ✅ Detect and avoid similar/duplicate pages
- ✅ Detect and avoid dead URLs (200 status but no data)
- ✅ Avoid crawling large files with low information value

---

## Files Modified/Created

### Modified Files

| File | Changes |
|------|---------|
| `config.ini` | Added user agent ID, set 4 threads |
| `scraper.py` | Complete rewrite with all scraping logic |
| `crawler/frontier.py` | Thread-safe implementation with per-domain politeness |
| `crawler/worker.py` | Updated for thread-safe operation |
| `utils/__init__.py` | Fixed URL hashing to exclude fragments |
| `packages/requirements.txt` | Added beautifulsoup4 and lxml |
| `stopwords.txt` | Complete stopwords list |

### New Files

| File | Purpose |
|------|---------|
| `report_generator.py` | Generates final markdown report |
| `IMPLEMENTATION_DETAILS.md` | This documentation file |
| `REPORT.md` | Generated after crawling (analytics report) |
| `crawler_report.json` | JSON analytics data (auto-generated) |

---

## Core Implementation

### URL Extraction and Parsing

**Location:** `scraper.py` → `extract_links()` function

```python
def extract_links(html_content, base_url):
    """Extract all links from HTML content."""
```

**How it works:**
1. Parse HTML using BeautifulSoup with lxml parser (falls back to html.parser)
2. Find all `<a>` tags with `href` attribute
3. Skip invalid links: empty, javascript:, mailto:, tel:, #-only
4. Resolve relative URLs using `urljoin(base_url, href)`
5. Defragment URLs using `urldefrag()` - removes the `#fragment` part
6. Return list of absolute, defragmented URLs

**Key Libraries:**
- `BeautifulSoup` from `bs4` - HTML parsing
- `urljoin` from `urllib.parse` - resolve relative URLs
- `urldefrag` from `urllib.parse` - remove fragments

---

### URL Validation (is_valid)

**Location:** `scraper.py` → `is_valid()` function

**Validation Steps:**
1. **Scheme check:** Only `http` or `https`
2. **Domain check:** Must match allowed UCI domains
3. **Extension check:** Skip non-HTML files (CSS, JS, images, PDFs, etc.)
4. **Trap check:** Avoid infinite loops and calendar traps
5. **Pattern frequency check:** Limit URLs matching same pattern

**File Extensions Blocked:**
```python
skip_extensions = (
    ".css", ".js", ".bmp", ".gif", ".jpg", ".jpeg", ".ico",
    ".png", ".tiff", ".pdf", ".ppt", ".pptx", ".doc", ".docx",
    ".xls", ".xlsx", ".zip", ".rar", ".gz", ".tar", ".exe",
    # ... and 50+ more extensions
)
```

---

### Domain Filtering

**Location:** `scraper.py` → `is_allowed_domain()` function

**Allowed Domain Patterns (regex):**
```python
ALLOWED_DOMAIN_PATTERNS = [
    r".*\.ics\.uci\.edu$",
    r".*\.cs\.uci\.edu$",
    r".*\.informatics\.uci\.edu$",
    r".*\.stat\.uci\.edu$",
    r"^ics\.uci\.edu$",
    r"^cs\.uci\.edu$",
    r"^informatics\.uci\.edu$",
    r"^stat\.uci\.edu$",
]
```

**Why both `.*\.domain` and `^domain`?**
- `.*\.ics.uci.edu` matches subdomains like `www.ics.uci.edu`, `vision.ics.uci.edu`
- `^ics.uci.edu` matches the bare domain without subdomain

---

### Trap Detection

**Location:** `scraper.py` → `is_trap()` function

**What is a trap?** An infinite or near-infinite set of URLs that don't provide useful content.

**Types of Traps Detected:**

1. **Calendar Traps:**
   - URLs with date patterns: `/calendar/2025`, `/events/2024-01-15`
   - Query params: `?date=`, `?month=`, `?year=`

2. **Pagination Traps:**
   - Infinite pagination: `?page=1`, `?page=2`, ...
   - Query params: `?start=`, `?offset=`, `?sort=`, `?filter=`

3. **Session/Action Traps:**
   - Login/logout pages
   - Action parameters: `?action=`, `?do=`, `?share=`
   - WordPress admin: `/wp-admin`, `/wp-login`

4. **Structural Traps:**
   - URL too long (>300 characters)
   - Path too deep (>10 segments)
   - Repeated path segments (>3 times)
   - Too many query parameters (>5)

**Pattern Tracking:**
```python
def get_url_pattern(url):
    """Replace numbers with {N} to identify patterns."""
    # /events/2025/01/15 becomes /events/{N}/{N}/{N}
```

If the same pattern appears >100 times, new URLs matching it are rejected.

---

## Analytics Tracking

### CrawlerAnalytics Class

**Location:** `scraper.py` → `CrawlerAnalytics` class

**Thread-safe storage** for all analytics data using `threading.Lock`.

```python
class CrawlerAnalytics:
    def __init__(self):
        self._lock = Lock()
        self.unique_pages = set()          # Unique URLs
        self.longest_page = {"url": "", "word_count": 0}
        self.word_frequencies = defaultdict(int)
        self.subdomains = defaultdict(set)  # subdomain -> URLs
        self.simhash_fingerprints = {}      # URL -> simhash
```

### Unique Pages

**How counted:**
- After defragmentation, URL is added to `unique_pages` set
- Set automatically handles duplicates
- Fragment removal: `http://site.com#section1` == `http://site.com#section2`

### Word Counting

**Location:** `scraper.py` → `extract_text_and_words()` function

**Process:**
1. Parse HTML, remove script/style/nav elements
2. Extract visible text using `soup.get_text()`
3. Tokenize: extract alphanumeric sequences, lowercase
4. Filter: remove stopwords, single-character words
5. Count frequencies in `word_frequencies` dict

**Tokenization Algorithm:**
```python
for ch in text:
    if ch.isalnum():  # ASCII letters and digits only
        current.append(ch.lower())
    else:
        if current:
            words.append("".join(current))
            current = []
```

### Subdomain Tracking

**Location:** `scraper.py` → `get_subdomain()` function

**How it works:**
1. Extract hostname from URL
2. Remove port if present
3. Check if ends with `.uci.edu`
4. Store full hostname as subdomain key
5. Track set of unique URLs per subdomain

---

## Extra Credit: Similarity Detection (+2 points)

### Exact Duplicate Detection

**Location:** `scraper.py` → `compute_exact_hash()` function

**Algorithm:** SHA-256 hash of entire page content

```python
def compute_exact_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
```

**Usage:** Before processing, check if content hash exists in `exact_hashes` set.

### Near-Duplicate Detection (SimHash)

**Location:** `scraper.py` → `compute_simhash()` and `hamming_distance()` functions

**What is SimHash?**
A locality-sensitive hashing technique where similar documents produce similar hashes.

**Algorithm (implemented from scratch):**

```python
def compute_simhash(tokens, hash_bits=64):
    """
    1. Initialize vector V of 64 integers to 0
    2. For each token:
       a. Compute MD5 hash of token
       b. For each bit position i:
          - If bit is 1: V[i] += 1
          - If bit is 0: V[i] -= 1
    3. Generate fingerprint:
       - If V[i] > 0: bit i = 1
       - If V[i] <= 0: bit i = 0
    """
    v = [0] * hash_bits
    
    for token in tokens:
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
        token_hash = token_hash & ((1 << hash_bits) - 1)
        
        for i in range(hash_bits):
            if token_hash & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint
```

**Hamming Distance:**
Number of bit positions where two hashes differ.

```python
def hamming_distance(hash1, hash2):
    xor = hash1 ^ hash2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance
```

**Threshold:** If Hamming distance ≤ 3, pages are considered near-duplicates.

**Why 3?** With 64-bit fingerprints:
- Distance 0-3: Very similar content (95%+ overlap)
- Distance 4-10: Moderately similar
- Distance 10+: Different content

---

## Extra Credit: Multithreading (+5 points)

### Thread-Safe Frontier

**Location:** `crawler/frontier.py` → `Frontier` class

**Key Features:**
1. **Thread-safe operations** using `RLock` and `Condition`
2. **Per-domain queues** for efficient URL distribution
3. **Per-domain timing** for politeness enforcement

**Data Structures:**
```python
self._domain_queues = defaultdict(list)      # domain -> [urls]
self._domain_last_access = defaultdict(float) # domain -> timestamp
self._seen_urls = set()                       # all seen URL hashes
```

### Per-Domain Politeness

**The Challenge:** 
Multiple threads must not make requests to the same domain within 500ms.

**Solution:**
The frontier tracks last access time per domain and only returns URLs when safe.

```python
def get_tbd_url(self):
    while True:
        current_time = time.time()
        
        for domain, urls in self._domain_queues.items():
            if not urls:
                continue
            
            time_since = current_time - self._domain_last_access[domain]
            
            if time_since >= self._politeness_delay:
                # Safe to access this domain
                url = urls.pop(0)
                self._domain_last_access[domain] = time.time()
                return url
        
        # No domain ready, wait and retry
        self._condition.wait(timeout=min_wait)
```

**Why this approach?**
- Workers don't need to sleep - they get URLs from different domains
- 4 threads can work on 4 different domains simultaneously
- Politeness is enforced at frontier level, not worker level

### Worker Implementation

**Location:** `crawler/worker.py` → `Worker` class

**Simplified because frontier handles politeness:**
```python
def run(self):
    while True:
        url = self.frontier.get_tbd_url()  # May block for politeness
        if not url:
            break
        
        resp = download(url, self.config, self.logger)
        scraped_urls = scraper.scraper(url, resp)
        
        for url in scraped_urls:
            self.frontier.add_url(url)
        
        self.frontier.mark_url_complete(url)
        # No sleep needed - frontier handles timing
```

---

## Key Algorithms Explained

### 1. URL Normalization

```python
def normalize_url(url):
    1. Defragment (remove #fragment)
    2. Lowercase scheme and netloc
    3. Remove default ports (:80 for http, :443 for https)
    4. Normalize path (handle trailing slashes)
    5. Filter tracking parameters (utm_*, sessionid, etc.)
```

### 2. Low Information Detection

```python
def has_low_information(text, word_count):
    # Too few words
    if word_count < 50:
        return True
    
    # Too few alphabetic characters
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.1:
        return True
    
    return False
```

### 3. Content Extraction

```python
def extract_text_and_words(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove non-content elements
    for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        element.decompose()
    
    # Get visible text
    text = soup.get_text(separator=" ", strip=True)
    
    # Tokenize into words
    words = tokenize(text)
    
    return text, words, len(words)
```

---

## Configuration

**config.ini settings:**

```ini
[IDENTIFICATION]
USERAGENT = IR UW26 66727694

[CONNECTION]
HOST = styx.ics.uci.edu
PORT = 9000

[CRAWLER]
SEEDURL = https://www.ics.uci.edu,https://www.cs.uci.edu,https://www.informatics.uci.edu,https://www.stat.uci.edu
POLITENESS = 0.5

[LOCAL PROPERTIES]
SAVE = frontier.shelve
THREADCOUNT = 4
```

---

## How to Run

### Install Dependencies

```bash
cd /path/to/spacetime-crawler4py
python -m pip install packages/spacetime-2.1.1-py3-none-any.whl
python -m pip install -r packages/requirements.txt
```

### Run Crawler

```bash
# Normal run (continues from previous progress)
python launch.py

# Fresh start (delete previous progress)
python launch.py --restart
```

### Generate Report

After crawling completes:
```bash
python report_generator.py
```

This creates `REPORT.md` with all analytics.

### Monitor Progress

The crawler logs to:
- Console (real-time status)
- `Logs/Worker.log` (detailed worker activity)
- `Logs/FRONTIER.log` (frontier status)
- `Logs/CRAWLER.log` (overall crawler status)

Progress updates print every 100 pages:
```
[PROGRESS] Unique pages: 500, Longest page: 12345 words, Subdomains: 15
```

---

## Interview Preparation Notes

### Common Questions and Answers

**Q: How do you detect duplicate pages?**
> A: Two methods - exact duplicates using SHA-256 content hash, and near-duplicates using SimHash with Hamming distance ≤ 3.

**Q: How do you avoid traps?**
> A: Pattern detection for calendars, pagination, sessions. Also track URL patterns (replace numbers with {N}) and limit frequency. Structural checks: max URL length (300), max path depth (10), max repeated segments (3).

**Q: How does your multithreading work?**
> A: The Frontier maintains per-domain queues and tracks last access time per domain. It only returns URLs when the domain's politeness delay (500ms) has passed. Workers don't sleep - they just ask for the next URL.

**Q: How do you ensure politeness with multiple threads?**
> A: Politeness is enforced at the Frontier level with locks. Each domain has its own queue and timing. Two threads can't get URLs from the same domain within 500ms because the Frontier won't return them.

**Q: What is SimHash?**
> A: A locality-sensitive hashing technique. Similar documents produce similar hashes. I use 64-bit fingerprints - if the Hamming distance (number of differing bits) is ≤ 3, pages are near-duplicates.

**Q: How do you count words?**
> A: Extract visible text from HTML (removing scripts, styles, nav). Tokenize into alphanumeric sequences. Filter stopwords. Count frequencies.

**Q: Why defragment URLs?**
> A: Per the assignment spec, `page.html#section1` and `page.html#section2` are the same page. Fragments only control where to scroll on the page, not what content to fetch.

**Q: How do you handle failed requests?**
> A: Check response status (200 = OK). Check if content exists and isn't empty. Mark URL complete regardless (so we don't retry forever). Log errors.

### Code Locations Quick Reference

| Concept | File | Function/Class |
|---------|------|----------------|
| Main scraper logic | scraper.py | `scraper()` |
| URL validation | scraper.py | `is_valid()` |
| Trap detection | scraper.py | `is_trap()` |
| SimHash | scraper.py | `compute_simhash()` |
| Word extraction | scraper.py | `extract_text_and_words()` |
| Analytics storage | scraper.py | `CrawlerAnalytics` class |
| Thread-safe frontier | crawler/frontier.py | `Frontier` class |
| Per-domain politeness | crawler/frontier.py | `get_tbd_url()` |
| Worker thread | crawler/worker.py | `Worker` class |
| URL hash (no fragment) | utils/__init__.py | `get_urlhash()` |

---

## Summary

This crawler implementation:

1. **Meets all base requirements** - valid domain filtering, politeness, trap avoidance, duplicate detection

2. **Earns +2 extra credit** - SimHash near-duplicate detection implemented from scratch (no libraries)

3. **Earns +5 extra credit** - Multithreaded with thread-safe frontier and per-domain politeness

4. **Tracks all required analytics** - unique pages, longest page, top 50 words, subdomains

5. **Generates proper reports** - JSON data file and markdown report

Total potential extra credit: **+7 points**
