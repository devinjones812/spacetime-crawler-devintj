"""
Web Scraper Module — URL extraction, validation, and analytics tracking.

Parses HTML pages, extracts links, tracks analytics for the assignment report,
detects traps, and performs near-duplicate detection via SimHash (extra credit).
"""

import atexit
import hashlib
import json
import os
import re
from collections import defaultdict
from threading import RLock
from urllib.parse import urlparse, urljoin, urldefrag

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Analytics Storage (thread-safe)
# ---------------------------------------------------------------------------

# Prune word-frequency dict when it exceeds MAX, keeping PRUNE_TO entries.
_WORD_FREQ_MAX = 1_000
_WORD_FREQ_PRUNE_TO = 500


class _SimHashIndex:
    """Bit-sampling index for O(1) average near-duplicate detection.

    Splits each 64-bit SimHash into 4 bands of 16 bits.  By pigeonhole,
    two hashes differing in <= 3 bits must share at least one identical
    band, so there are zero false negatives for threshold <= 3.
    """

    _NUM_BANDS = 4
    _BAND_BITS = 16  # 64 / 4

    def __init__(self):
        self._bands = [defaultdict(list) for _ in range(self._NUM_BANDS)]
        self._exact = set()  # short-circuit for exact matches

    def _get_bands(self, h):
        mask = (1 << self._BAND_BITS) - 1
        return [(h >> (i * self._BAND_BITS)) & mask for i in range(self._NUM_BANDS)]

    def contains_near(self, h, threshold=3):
        """Return True if *h* is within *threshold* hamming distance of any stored hash."""
        if h in self._exact:
            return True
        candidates = set()
        for i, bv in enumerate(self._get_bands(h)):
            for existing in self._bands[i].get(bv, ()):
                candidates.add(existing)
        for c in candidates:
            if _hamming_distance(h, c) <= threshold:
                return True
        return False

    def add(self, h):
        self._exact.add(h)
        for i, bv in enumerate(self._get_bands(h)):
            self._bands[i][bv].append(h)


class CrawlerAnalytics:
    """Thread-safe storage for crawler analytics data."""

    def __init__(self, resume_file="crawler_report.json"):
        self._lock = RLock()
        self.unique_page_count: int = 0
        self.longest_page = {"url": "", "word_count": 0}
        self.word_frequencies: defaultdict = defaultdict(int)
        self.subdomains: defaultdict = defaultdict(int)       # subdomain -> count
        self._simhash_index = _SimHashIndex()
        self.url_patterns: defaultdict = defaultdict(int)     # pattern -> count
        self.exact_hashes: set = set()                        # SHA-256 digests (bytes)
        self.stopwords: set = set()

        # Cumulative count from a previous run (for resume)
        self.previous_unique_count = 0
        self.subdomain_resumed_counts: dict = {}              # subdomain -> int

        self._load_stopwords()
        self._load_previous_report(resume_file)

    # -- persistence helpers ------------------------------------------------

    def _load_stopwords(self):
        path = os.path.join(os.path.dirname(__file__), "stopwords.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        self.stopwords.add(w)
                        self.stopwords.add(w.replace("'", ""))
        except FileNotFoundError:
            self.stopwords = {
                "a", "an", "and", "the", "to", "of", "in", "is", "it", "for"
            }

    def _load_previous_report(self, filepath):
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.longest_page = data.get("longest_page", self.longest_page)
            word_data = data.get(
                "word_frequencies_for_resume",
                data.get("all_word_frequencies", data.get("top_50_words", [])),
            )
            for word, count in word_data:
                self.word_frequencies[word] = count
            for subdomain, count in data.get("subdomains", []):
                self.subdomain_resumed_counts[subdomain] = count
            self.previous_unique_count = data.get("unique_pages_count", 0)
            print(f"[ANALYTICS] Resumed: {self.previous_unique_count} pages, "
                  f"{len(self.word_frequencies)} words")
        except Exception as e:
            print(f"[ANALYTICS] Could not load previous report: {e}")

    # -- recording ----------------------------------------------------------

    def add_page(self, url, word_count, words, subdomain):
        """Record a page's analytics data."""
        with self._lock:
            self.unique_page_count += 1

            if word_count > self.longest_page["word_count"]:
                self.longest_page = {"url": url, "word_count": word_count}

            for w in words:
                if w.isalpha() and len(w) > 1 and w not in self.stopwords:
                    self.word_frequencies[w] += 1
            self._prune_word_frequencies()

            if subdomain:
                self.subdomains[subdomain] += 1

    def is_exact_duplicate(self, content_digest):
        """Return True if this content digest (bytes) has been seen before."""
        with self._lock:
            if content_digest in self.exact_hashes:
                return True
            self.exact_hashes.add(content_digest)
            return False

    def is_near_duplicate(self, simhash_value, threshold=3):
        """Check if page is near-duplicate using SimHash hamming distance."""
        with self._lock:
            return self._simhash_index.contains_near(simhash_value, threshold)

    def add_simhash(self, simhash_value):
        with self._lock:
            self._simhash_index.add(simhash_value)

    def record_url_pattern(self, pattern):
        """Increment and return the count for *pattern*."""
        with self._lock:
            self.url_patterns[pattern] += 1
            return self.url_patterns[pattern]

    # -- queries ------------------------------------------------------------

    def get_top_words(self, n=50):
        with self._lock:
            return sorted(
                self.word_frequencies.items(),
                key=lambda x: (-x[1], x[0]),
            )[:n]

    def get_subdomain_stats(self):
        with self._lock:
            merged = dict(self.subdomain_resumed_counts)
            for sub, count in self.subdomains.items():
                merged[sub] = merged.get(sub, 0) + count
            return sorted(merged.items())

    def get_unique_count(self):
        with self._lock:
            return self.previous_unique_count + self.unique_page_count

    # -- report persistence -------------------------------------------------

    def save_report(self, filepath="crawler_report.json"):
        with self._lock:
            all_words = sorted(
                self.word_frequencies.items(), key=lambda x: (-x[1], x[0])
            )
            report = {
                "unique_pages_count": self.get_unique_count(),
                "longest_page": self.longest_page,
                "top_50_words": all_words[:50],
                "word_frequencies_for_resume": all_words,
                "subdomains": self.get_subdomain_stats(),
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # -- internal -----------------------------------------------------------

    def _prune_word_frequencies(self):
        """Keep the dict bounded — must be called while holding _lock."""
        if len(self.word_frequencies) > _WORD_FREQ_MAX:
            top = sorted(self.word_frequencies.items(), key=lambda x: -x[1])[
                :_WORD_FREQ_PRUNE_TO
            ]
            self.word_frequencies.clear()
            self.word_frequencies.update(top)


# Global analytics instance
analytics = CrawlerAnalytics()


# ---------------------------------------------------------------------------
# SimHash (Extra Credit — implemented from scratch)
# ---------------------------------------------------------------------------

def compute_simhash(tokens, hash_bits=64):
    """Compute a SimHash fingerprint for a list of tokens."""
    if not tokens:
        return 0
    v = [0] * hash_bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        h &= (1 << hash_bits) - 1
        for i in range(hash_bits):
            v[i] += 1 if (h & (1 << i)) else -1
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def _hamming_distance(a, b):
    """Number of differing bits between two integers."""
    x = a ^ b
    d = 0
    while x:
        d += x & 1
        x >>= 1
    return d


# ---------------------------------------------------------------------------
# HTML Parsing (shared soup for link extraction + text extraction)
# ---------------------------------------------------------------------------

def _parse_html(content):
    """Parse HTML content into a BeautifulSoup object. Returns None on failure."""
    try:
        return BeautifulSoup(content, "lxml")
    except Exception:
        try:
            return BeautifulSoup(content, "html.parser")
        except Exception:
            return None


def extract_links(soup, base_url):
    """Extract and defragment all <a href> links from a parsed soup."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        try:
            absolute = urljoin(base_url, href)
            defragged, _ = urldefrag(absolute)
            links.append(defragged)
        except Exception:
            continue
    return links


def extract_text_and_words(soup):
    """Return (raw_text, word_list, word_count) from a parsed soup.

    Operates on a *copy* so decompose() doesn't affect link extraction
    when both functions share the same original soup.
    """
    soup = soup.__copy__()
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Tokenize: extract alphanumeric runs, lowercase
    words = re.findall(r"[a-zA-Z0-9]+", text)
    words = [w.lower() for w in words]
    return text, words, len(words)


def _has_low_information(text, word_count, min_words=50, min_alpha_ratio=0.1):
    """True if the page has too little textual content."""
    if word_count < min_words:
        return True
    if text:
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / len(text) < min_alpha_ratio:
            return True
    return False


_SOFT_404_PHRASES = [
    "page not found", "404", "not found", "does not exist",
    "no longer available", "has been removed", "has been deleted",
    "cannot be found", "could not be found",
]


def _is_soft_404(text):
    """Detect soft-404 pages (status 200 but actually an error page)."""
    if len(text) > 1000:
        return False
    t = text.lower()
    return sum(1 for p in _SOFT_404_PHRASES if p in t) >= 2


# ---------------------------------------------------------------------------
# URL Helpers
# ---------------------------------------------------------------------------

def get_subdomain(url):
    """Return the hostname if it's a *.uci.edu subdomain, else None."""
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
        return host if host.endswith(".uci.edu") or host == "uci.edu" else None
    except Exception:
        return None


def normalize_url(url):
    """Normalize a URL for deduplication (lowercase host, defrag, strip trailing slash)."""
    try:
        url, _ = urldefrag(url)
        p = urlparse(url)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        path = p.path if p.path else "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Strip common tracking query parameters
        query = p.query
        if query:
            tracking = {
                "utm_source", "utm_medium", "utm_campaign", "utm_term",
                "utm_content", "sessionid", "sid", "phpsessid", "jsessionid",
            }
            parts = [
                param for param in query.split("&")
                if param.split("=")[0].lower() not in tracking
            ]
            query = "&".join(parts)

        result = f"{scheme}://{netloc}{path}"
        if query:
            result += f"?{query}"
        return result
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Trap Detection
# ---------------------------------------------------------------------------

TRAP_PATTERNS = [
    # GitLab internals
    r"/-/",
    # Photo / image gallery traps
    r"/pix/.+\.html$",
    r"/photos?/.+\.html$",
    r"/gallery/.+\.html$",
    r"/images?/.+\.html$",
    r"/(spring|summer|fall|winter)\d{2}\.html$",
    # Course slides / presentations
    r"/slides?/node\d+\.html?$",
    r"/slides?/slide\d+\.html?$",
    r"/slides?/img\d+\.html?$",
    r"/lectures?/.+/node\d+\.html?$",
    r"/presentations?/.+/tsld\d+\.htm",
    r"/presentations?/.+/sld\d+\.htm",
    r"/sld\d+\.htm",
    r"/tsld\d+\.htm",
    r"/slide\d+\.htm",
    r"/wisen/wisen\d+/",
    # Calendar / date traps
    r"/calendar[/\?].*\d{4}",
    r"/events?[/\?].*\d{4}",
    r"[?&]date=",
    r"[?&]month=",
    r"[?&]year=",
    r"[?&]day=",
    # Pagination / sorting
    r"[?&]page=\d+",
    r"[?&]start=\d+",
    r"[?&]offset=\d+",
    r"[?&]sort=",
    r"[?&]order=",
    r"[?&]filter=",
    # Session / action / auth
    r"[?&]action=",
    r"[?&]do=",
    r"[?&]share=",
    r"[?&]replytocom=",
    r"[?&]login",
    r"/login",
    r"/logout",
    r"/wp-admin",
    r"/wp-login",
    # Dynamic / diff views
    r"[?&]rev=",
    r"[?&]version=",
    r"[?&]diff=",
    # Apache directory listing sort params
    r"[?&]C=[NMSD]",
    # Trac raw attachments
    r"/raw-attachment/",
    # DokuWiki traps
    r"/doku\.php",
    r"[?&]idx=",
    # Calendar export downloads
    r"[?&]ical=",
    r"[?&]outlook-ical=",
    # Trac timeline with timestamps
    r"/timeline\?",
    # Raw text wiki exports
    r"[?&]format=txt",
]

_COMPILED_TRAPS = [re.compile(p, re.IGNORECASE) for p in TRAP_PATTERNS]

MAX_URL_LENGTH = 300
MAX_PATH_DEPTH = 10
MAX_REPEATED_SEGMENTS = 3
MAX_CONTENT_SIZE = 5_000_000  # 5 MB

# Pattern-frequency thresholds for trap detection
MAX_FINE_PATTERN_COUNT = 50
MAX_COARSE_PATTERN_COUNT = 200

_LOW_VALUE_PATHS = [
    "/wp-content/", "/wp-includes/", "/assets/", "/static/",
    "/_includes/", "/templates/", "/print/", "/mobile/",
    "/feed/", "/feeds/", "/xmlrpc.php", "/trackback/",
]

_SKIP_EXTENSIONS = frozenset([
    ".css", ".js", ".bmp", ".gif", ".jpg", ".jpeg", ".ico", ".png",
    ".tiff", ".tif", ".mid", ".mp2", ".mp3", ".mp4", ".wav", ".avi",
    ".mov", ".mpeg", ".ram", ".m4v", ".mkv", ".ogg", ".ogv", ".pdf",
    ".ps", ".eps", ".tex", ".ppt", ".pptx", ".doc", ".docx", ".xls",
    ".xlsx", ".names", ".data", ".dat", ".exe", ".bz2", ".tar", ".msi",
    ".bin", ".7z", ".psd", ".dmg", ".iso", ".epub", ".dll", ".cnf",
    ".tgz", ".sha1", ".thmx", ".mso", ".arff", ".rtf", ".jar", ".csv",
    ".rm", ".smil", ".wmv", ".swf", ".wma", ".zip", ".rar", ".gz",
    ".img", ".sql", ".db", ".sqlite", ".json", ".xml", ".rss", ".atom",
    ".apk", ".war", ".ear", ".class", ".pyc", ".pyo", ".so", ".o",
    ".a", ".lib", ".deb", ".rpm", ".pkg", ".mpg", ".flv", ".webm",
    ".svg", ".ttf", ".woff", ".woff2", ".eot", ".otf", ".bak", ".tmp",
    ".log", ".out", ".mat", ".m", ".r", ".ipynb", ".nb", ".ss", ".ppsx",
    ".cc", ".h", ".hpp", ".cpp", ".c", ".java", ".py", ".pl", ".sh",
    ".scm", ".rkt", ".odc", ".conf", ".dsw", ".dsp", ".inc", ".sas",
    ".fig", ".cls", ".tsv", ".txt",
])

# Allowed domains per the assignment spec
_ALLOWED_DOMAINS = [
    re.compile(p, re.IGNORECASE) for p in [
        r".*\.ics\.uci\.edu$",
        r".*\.cs\.uci\.edu$",
        r".*\.informatics\.uci\.edu$",
        r".*\.stat\.uci\.edu$",
        r"^ics\.uci\.edu$",
        r"^cs\.uci\.edu$",
        r"^informatics\.uci\.edu$",
        r"^stat\.uci\.edu$",
    ]
]


def _is_allowed_domain(hostname):
    hostname = hostname.lower().split(":")[0]
    return any(p.match(hostname) for p in _ALLOWED_DOMAINS)


def _is_trap(url):
    """Return True if the URL looks like an infinite trap."""
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        full = path + ("?" + query if query else "")

        if len(url) > MAX_URL_LENGTH:
            return True

        segments = [s for s in path.split("/") if s]
        if len(segments) > MAX_PATH_DEPTH:
            return True

        counts: dict = {}
        for seg in segments:
            counts[seg] = counts.get(seg, 0) + 1
            if counts[seg] > MAX_REPEATED_SEGMENTS:
                return True

        for pat in _COMPILED_TRAPS:
            if pat.search(full):
                return True

        if query and query.count("=") > 5:
            return True

        return False
    except Exception:
        return True


def _get_url_patterns(url):
    """Return (fine_pattern, coarse_pattern) for pattern-frequency trap detection."""
    try:
        parsed = urlparse(url)
        fine = re.sub(r"\d+", "{N}", parsed.path)
        parts = [p for p in parsed.path.split("/") if p]
        coarse = ("/" + "/".join(parts[:3]) + "/*") if len(parts) > 3 else parsed.path
        return f"{parsed.netloc}{fine}", f"{parsed.netloc}{coarse}"
    except Exception:
        return url, url


def _get_extension(path_lower):
    """Extract file extension via rsplit for O(1) set lookup."""
    dot = path_lower.rfind(".")
    return path_lower[dot:] if dot != -1 else ""


# ---------------------------------------------------------------------------
# is_valid — the single entry point the frontier calls
# ---------------------------------------------------------------------------

def is_valid(url):
    """Return True if *url* should be crawled."""
    try:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return False

        if not _is_allowed_domain(parsed.netloc):
            return False

        path_lower = parsed.path.lower()

        # Low-value path prefixes
        if any(path_lower.startswith(p) for p in _LOW_VALUE_PATHS):
            return False

        # Non-HTML file extensions — O(1) set lookup
        if _get_extension(path_lower) in _SKIP_EXTENSIONS:
            return False

        if _is_trap(url):
            return False

        # Pattern-frequency limits
        fine, coarse = _get_url_patterns(url)
        if analytics.record_url_pattern(fine) > MAX_FINE_PATTERN_COUNT:
            return False
        if MAX_COARSE_PATTERN_COUNT > 0:
            if analytics.record_url_pattern(coarse) > MAX_COARSE_PATTERN_COUNT:
                return False

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main scraper function
# ---------------------------------------------------------------------------

def scraper(url, resp):
    """
    Parse a downloaded page, record analytics, and return new URLs to crawl.

    Args:
        url:  the URL that was fetched
        resp: Response object (status, raw_response, …)

    Returns:
        list of URLs extracted from the page
    """
    url = normalize_url(url)

    if resp.status != 200:
        return []
    if not resp.raw_response or not resp.raw_response.content:
        return []

    try:
        content = resp.raw_response.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
    except Exception:
        return []

    if len(content) < 100 or len(content) > MAX_CONTENT_SIZE:
        return []

    # Parse HTML once — shared between link extraction and text extraction
    soup = _parse_html(content)
    if soup is None:
        return []

    # Extract links (returned regardless of analytics decisions)
    base = resp.raw_response.url if resp.raw_response else url
    valid_links = [link for link in extract_links(soup, base) if is_valid(link)]

    # --- analytics gate: skip recording if content is duplicate / low-value ---

    content_digest = hashlib.sha256(content.encode("utf-8")).digest()
    if analytics.is_exact_duplicate(content_digest):
        return valid_links

    text, words, word_count = extract_text_and_words(soup)

    if _is_soft_404(text):
        return valid_links
    if _has_low_information(text, word_count):
        return valid_links

    sh = compute_simhash(words)
    if analytics.is_near_duplicate(sh):
        return valid_links
    analytics.add_simhash(sh)

    analytics.add_page(url, word_count, words, get_subdomain(url))

    # Periodic save for crash recovery
    count = analytics.get_unique_count()
    if count % 50 == 0:
        analytics.save_report()
    if count % 100 == 0:
        print(f"[PROGRESS] {count} unique pages | "
              f"longest: {analytics.longest_page['word_count']} words")

    return valid_links


# ---------------------------------------------------------------------------
# Final report (printed on exit)
# ---------------------------------------------------------------------------

def print_final_report():
    print("\n" + "=" * 70)
    print("CRAWLER ANALYTICS REPORT")
    print("=" * 70)

    count = analytics.get_unique_count()
    print(f"\n1. UNIQUE PAGES: {count}")

    lp = analytics.longest_page
    print(f"\n2. LONGEST PAGE:\n   URL: {lp['url']}\n   Words: {lp['word_count']}")

    print("\n3. TOP 50 MOST COMMON WORDS:")
    for i, (word, freq) in enumerate(analytics.get_top_words(50), 1):
        print(f"   {i:2}. {word}: {freq}")

    subs = analytics.get_subdomain_stats()
    print(f"\n4. SUBDOMAINS ({len(subs)} total):")
    for sub, n in subs:
        print(f"   {sub}, {n}")

    print("=" * 70)

    analytics.save_report()
    print("Report saved to crawler_report.json")


atexit.register(print_final_report)
