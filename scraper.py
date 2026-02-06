"""
Web Scraper Module - Implements URL extraction, validation, and analytics tracking.

This module provides the core scraping functionality for the web crawler:
- HTML parsing and link extraction using BeautifulSoup
- URL normalization and defragmentation
- Domain validation for allowed UCI domains
- Analytics tracking for report generation
- Trap detection (infinite URL patterns, calendars, etc.)
- Similarity detection using SimHash (extra credit)
"""

import re
import os
import json
import hashlib
from urllib.parse import urlparse, urljoin, urldefrag
from collections import defaultdict
from threading import Lock, RLock
from bs4 import BeautifulSoup

# Enable link filtering diagnostics with: CRAWLER_DEBUG=1
CRAWLER_DEBUG = os.environ.get("CRAWLER_DEBUG") == "1"
MAX_DEBUG_INVALID = 10

# ============================================================================
# ANALYTICS STORAGE (Thread-safe)
# ============================================================================

class CrawlerAnalytics:
    """Thread-safe storage for crawler analytics data."""

    def __init__(self, resume_file="crawler_report.json"):
        # Re-entrant lock prevents deadlocks when helper methods
        # call other methods that also acquire the lock.
        self._lock = RLock()
        self.unique_pages = set()  # Set of unique URLs (defragmented)
        self.longest_page = {"url": "", "word_count": 0}
        self.word_frequencies = defaultdict(int)  # word -> count
        self.subdomains = defaultdict(set)  # subdomain -> set of URLs
        self.simhash_fingerprints = {}  # url -> simhash value
        self.url_patterns = defaultdict(int)  # pattern -> count (for trap detection)
        self.stopwords = set()

        # Track cumulative count separately for accurate resumption
        self.previous_unique_count = 0

        # Trap tracking (NEW!)
        self.traps_detected = defaultdict(lambda: {"count": 0, "examples": []})
        self.trap_patterns_triggered = defaultdict(int)  # pattern -> how many times hit limit

        self._load_stopwords()
        self._load_previous_report(resume_file)

    def _load_previous_report(self, filepath):
        """Load previous analytics to resume from where we left off."""
        if not os.path.exists(filepath):
            print(f"[ANALYTICS] No previous report found at {filepath}, starting fresh")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Restore longest page
            self.longest_page = data.get("longest_page", {"url": "", "word_count": 0})

            # Restore word frequencies - use ALL words if available, fallback to top 50
            word_data = data.get("all_word_frequencies", data.get("top_50_words", []))
            for word, count in word_data:
                self.word_frequencies[word] = count

            # Restore subdomains (can't restore exact URLs, but can track counts)
            for subdomain, count in data.get("subdomains", []):
                # Create dummy entries to maintain count
                self.subdomains[subdomain] = set([f"{subdomain}_page_{i}" for i in range(count)])

            self.previous_unique_count = data.get("unique_pages_count", 0)
            print(f"[ANALYTICS] Loaded previous report: {self.previous_unique_count} pages, "
                  f"{len(self.word_frequencies)} words, {len(self.subdomains)} subdomains")
            print(f"[ANALYTICS] Resuming analytics - new pages will add to this count")
            print(f"[ANALYTICS] Total unique pages will be: {self.previous_unique_count} + new pages")

        except Exception as e:
            print(f"[ANALYTICS] Error loading previous report: {e}, starting fresh")

    def reset(self):
        """Reset all analytics data for a clean restart."""
        with self._lock:
            self.unique_pages.clear()
            self.longest_page = {"url": "", "word_count": 0}
            self.word_frequencies.clear()
            self.subdomains.clear()
            self.simhash_fingerprints.clear()
            self.url_patterns.clear()
            self.previous_unique_count = 0
            self.traps_detected.clear()
            self.trap_patterns_triggered.clear()
            print("[ANALYTICS] Reset all analytics data for fresh start")

    def _load_stopwords(self):
        """Load stopwords from file."""
        stopwords_path = os.path.join(os.path.dirname(__file__), "stopwords.txt")
        try:
            with open(stopwords_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        # Handle contractions - store both forms
                        self.stopwords.add(word)
                        # Also add version without apostrophe
                        self.stopwords.add(word.replace("'", ""))
        except FileNotFoundError:
            print("[WARNING] stopwords.txt not found, using minimal default stopwords")
            # Minimal fallback if stopwords.txt is missing
            self.stopwords = {"a", "an", "and", "the", "to", "of", "in", "is", "it", "for"}
    
    def add_page(self, url, word_count, words, subdomain):
        """Record a page's analytics data."""
        with self._lock:
            # Track unique page
            self.unique_pages.add(url)
            
            # Track longest page
            if word_count > self.longest_page["word_count"]:
                self.longest_page = {"url": url, "word_count": word_count}
            
            # Track word frequencies (excluding stopwords)
            for word in words:
                if word.isalpha() and len(word) > 1 and word not in self.stopwords:
                    self.word_frequencies[word] += 1
            
            # Track subdomain
            if subdomain:
                self.subdomains[subdomain].add(url)
    
    def is_near_duplicate(self, simhash_value, threshold=3):
        """Check if page is near-duplicate using SimHash hamming distance."""
        with self._lock:
            for existing_hash in self.simhash_fingerprints.values():
                if hamming_distance(simhash_value, existing_hash) <= threshold:
                    return True
            return False
    
    def add_simhash(self, url, simhash_value):
        """Store a page's SimHash fingerprint."""
        with self._lock:
            self.simhash_fingerprints[url] = simhash_value
    
    def record_url_pattern(self, pattern):
        """Record URL pattern for trap detection."""
        with self._lock:
            self.url_patterns[pattern] += 1
            return self.url_patterns[pattern]
    
    def get_top_words(self, n=50):
        """Get top N most common words."""
        with self._lock:
            sorted_words = sorted(
                self.word_frequencies.items(),
                key=lambda x: (-x[1], x[0])
            )
            return sorted_words[:n]
    
    def get_subdomain_stats(self):
        """Get subdomain statistics sorted alphabetically."""
        with self._lock:
            return sorted(
                [(subdomain, len(urls)) for subdomain, urls in self.subdomains.items()],
                key=lambda x: x[0]
            )
    
    def get_stats(self):
        """Get current analytics statistics."""
        with self._lock:
            return {
                "unique_pages": self.previous_unique_count + len(self.unique_pages),
                "longest_page": self.longest_page.copy(),
                "total_words_tracked": len(self.word_frequencies),
                "total_subdomains": len(self.subdomains)
            }
    
    def record_trap(self, url, trap_type, pattern=None):
        """Record a URL that was blocked as a trap."""
        with self._lock:
            trap_entry = self.traps_detected[trap_type]
            trap_entry["count"] += 1

            # Save up to 10 examples per trap type
            if len(trap_entry["examples"]) < 10:
                example = {"url": url}
                if pattern:
                    example["pattern"] = pattern
                trap_entry["examples"].append(example)

            # Track pattern frequency traps
            if pattern and "pattern_frequency" in trap_type:
                self.trap_patterns_triggered[pattern] += 1

    def save_report(self, filepath="crawler_report.json"):
        """Save analytics to a JSON file."""
        with self._lock:
            # Include cumulative count from previous runs
            total_unique = self.previous_unique_count + len(self.unique_pages)

            # Save ALL word frequencies for accurate resumption (not just top 50)
            all_words = sorted(self.word_frequencies.items(), key=lambda x: (-x[1], x[0]))

            report = {
                "unique_pages_count": total_unique,
                "longest_page": self.longest_page,
                "top_50_words": all_words[:50],  # Top 50 for the report
                "all_word_frequencies": all_words,  # ALL words for resumption
                "subdomains": self.get_subdomain_stats()
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

    def save_trap_report(self, filepath="trap_report.json"):
        """Save trap detection report for analysis."""
        with self._lock:
            # Convert defaultdict to regular dict for JSON
            traps_dict = {}
            total_blocked = 0
            for trap_type, data in self.traps_detected.items():
                traps_dict[trap_type] = {
                    "count": data["count"],
                    "examples": data["examples"]
                }
                total_blocked += data["count"]

            report = {
                "summary": {
                    "total_urls_blocked": total_blocked,
                    "trap_types_detected": len(traps_dict),
                    "top_patterns_blocked": sorted(
                        self.trap_patterns_triggered.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:20]
                },
                "trap_details": traps_dict
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            print(f"\n[TRAP REPORT] Blocked {total_blocked} URLs across {len(traps_dict)} trap types")
            print(f"[TRAP REPORT] Report saved to {filepath}")


# Global analytics instance
analytics = CrawlerAnalytics()


# ============================================================================
# SIMHASH IMPLEMENTATION (Extra Credit - from scratch)
# ============================================================================

def compute_simhash(tokens, hash_bits=64):
    """
    Compute SimHash fingerprint for a list of tokens.
    
    SimHash algorithm:
    1. Initialize a vector V of hash_bits integers to 0
    2. For each token, compute its hash
    3. For each bit position i in the hash:
       - If bit i is 1, add weight to V[i]
       - If bit i is 0, subtract weight from V[i]
    4. Final fingerprint: V[i] > 0 -> bit i is 1, else 0
    """
    if not tokens:
        return 0
    
    v = [0] * hash_bits
    
    for token in tokens:
        # Compute hash of token
        token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        # Use only hash_bits bits
        token_hash = token_hash & ((1 << hash_bits) - 1)
        
        # Update vector based on bits
        for i in range(hash_bits):
            if token_hash & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    
    # Generate fingerprint
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint


def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two hashes (number of differing bits)."""
    xor = hash1 ^ hash2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance


def compute_exact_hash(content):
    """Compute exact hash of content for duplicate detection."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# ============================================================================
# TEXT EXTRACTION AND TOKENIZATION
# ============================================================================

def extract_text_and_words(html_content):
    """
    Extract text from HTML and tokenize into words.
    Returns: (raw_text, list_of_words, word_count)
    """
    if not html_content:
        return "", [], 0
    
    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception:
            return "", [], 0
    
    # Remove script and style elements
    for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        element.decompose()
    
    # Get text
    text = soup.get_text(separator=" ", strip=True)
    
    # Tokenize - extract alphanumeric sequences
    words = []
    current = []
    for ch in text:
        code = ord(ch)
        if (48 <= code <= 57 or  # 0-9
            65 <= code <= 90 or  # A-Z
            97 <= code <= 122):  # a-z
            current.append(ch.lower())
        else:
            if current:
                words.append("".join(current))
                current = []
    if current:
        words.append("".join(current))
    
    return text, words, len(words)


def has_low_information(text, word_count, threshold_words=50, threshold_ratio=0.1):
    """
    Detect pages with low textual information content.
    Returns True if page should be skipped.
    """
    # Too few words
    if word_count < threshold_words:
        return True

    # Check text to content ratio
    if len(text) > 0:
        # If the text is mostly numbers or single characters
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / len(text) < threshold_ratio:
            return True

    return False


def is_soft_404(text):
    """
    Detect soft 404s - pages that return 200 but are actually error pages.
    Returns True if page appears to be an error page.
    """
    if len(text) > 1000:
        # Only check short pages (real 404s are usually brief)
        return False

    text_lower = text.lower()
    # Count how many indicators are present
    indicator_count = sum(1 for indicator in SOFT_404_INDICATORS if indicator in text_lower)

    # If multiple indicators in a short page, likely a soft 404
    return indicator_count >= 2


def is_low_value_path(url):
    """
    Check if URL path is known to contain low-value content.
    Returns True if path should be skipped.
    """
    try:
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        return any(path_lower.startswith(prefix) for prefix in LOW_VALUE_PATHS)
    except Exception:
        return False


# ============================================================================
# URL EXTRACTION AND NORMALIZATION
# ============================================================================

def extract_links(html_content, base_url):
    """Extract all links from HTML content."""
    links = []
    
    if not html_content:
        return links
    
    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception:
            return links
    
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        
        # Skip empty, javascript, mailto links
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        
        # Resolve relative URLs
        try:
            absolute_url = urljoin(base_url, href)
            # Defragment the URL
            defragged_url, _ = urldefrag(absolute_url)
            links.append(defragged_url)
        except Exception:
            continue
    
    return links


def get_subdomain(url):
    """Extract subdomain from URL for uci.edu domain tracking."""
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        
        # Remove port if present
        if ":" in hostname:
            hostname = hostname.split(":")[0]
        
        # Check if it's a uci.edu domain
        if hostname.endswith(".uci.edu") or hostname == "uci.edu":
            return hostname
        
        return None
    except Exception:
        return None


def normalize_url(url):
    """Normalize URL for comparison."""
    try:
        # Defragment
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        
        # Lowercase scheme and netloc
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        
        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        
        # Normalize path - remove trailing slash, handle empty path
        path = parsed.path
        if path == "" or path == "/":
            path = "/"
        elif path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")
        
        # Remove common session/tracking parameters
        query = parsed.query
        if query:
            # Filter out common tracking params
            tracking_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                            "utm_content", "sessionid", "sid", "PHPSESSID", "jsessionid"}
            params = query.split("&")
            filtered = [p for p in params if p.split("=")[0].lower() not in tracking_params]
            query = "&".join(filtered)
        
        # Reconstruct URL
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        
        return normalized
    except Exception:
        return url


# ============================================================================
# TRAP DETECTION
# ============================================================================

# Patterns that indicate potential traps
TRAP_PATTERNS = [
    # GitLab repository traps (commits, trees, blobs — effectively infinite)
    r"/-/",

    # Photo gallery traps (like eppstein/pix)
    r"/pix/.+\.html$",  # Individual photo pages
    r"/photos?/.+\.html$",
    r"/gallery/.+\.html$",
    r"/images?/.+\.html$",
    r"/(spring|summer|fall|winter)\d{2}\.html$",  # Seasonal galleries

    # Course material traps (like dechter slides)
    r"/slides?/node\d+\.html?$",  # Slide pagination (node1, node2, etc.)
    r"/slides?/slide\d+\.html?$",  # Alternative slide format
    r"/slides?/img\d+\.html?$",   # Image slides
    r"/lectures?/.+/node\d+\.html?$",  # Lecture slides
    r"/presentations?/.+/tsld\d+\.htm",  # Presentation slides (tsld001, tsld002, etc.)
    r"/presentations?/.+/sld\d+\.htm",   # Presentation slides (sld001, sld002, etc.)
    r"/wisen/wisen\d+/",  # WISEN conference presentations (major trap)

    # Calendar traps
    r"/calendar[/\?].*\d{4}",
    r"/events?[/\?].*\d{4}",
    r"[?&]date=",
    r"[?&]month=",
    r"[?&]year=",
    r"[?&]day=",

    # Pagination/sorting traps
    r"[?&]page=\d+",
    r"[?&]start=\d+",
    r"[?&]offset=\d+",
    r"[?&]sort=",
    r"[?&]order=",
    r"[?&]filter=",
    
    # Session/action traps
    r"[?&]action=",
    r"[?&]do=",
    r"[?&]share=",
    r"[?&]replytocom=",
    r"[?&]login",
    r"/login",
    r"/logout",
    r"/wp-admin",
    r"/wp-login",
    
    # Dynamic content traps
    r"[?&]rev=",
    r"[?&]version=",
    r"[?&]diff=",

    # Apache directory listing sort parameters (duplicate views of same listing)
    r"[?&]C=[NMSD]",

    # Trac wiki raw file attachments (binary downloads, not web pages)
    r"/raw-attachment/",

    # DokuWiki traps (namespace browser creates combinatorial explosion)
    r"/doku\.php",

    # DokuWiki namespace index (duplicate sidebar content per page)
    r"[?&]idx=",

    # Calendar export downloads (not HTML pages)
    r"[?&]ical=",
    r"[?&]outlook-ical=",

    # Trac timeline with timestamps (infinite date variations)
    r"/timeline\?",

    # Raw text wiki exports (duplicate of HTML version)
    r"[?&]format=txt",
]

# Compiled trap patterns for efficiency
COMPILED_TRAP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TRAP_PATTERNS]

# Maximum URL length
MAX_URL_LENGTH = 300

# Maximum depth in path
MAX_PATH_DEPTH = 10

# Maximum repeated path segments
MAX_REPEATED_SEGMENTS = 3

# Maximum content size (bytes) - skip very large files
MAX_CONTENT_SIZE = 5_000_000  # 5 MB

# ============================================================================
# LOW-VALUE CONTENT DETECTION
# ============================================================================

# Path prefixes known to contain low-value content
LOW_VALUE_PATHS = [
    "/wp-content/",      # WordPress assets
    "/wp-includes/",     # WordPress system files
    "/assets/",          # Asset directories
    "/static/",          # Static file directories
    "/_includes/",       # Template includes
    "/templates/",       # Template directories
    "/print/",           # Print versions (duplicate content)
    "/mobile/",          # Mobile versions (duplicate content)
    "/feed/",            # RSS feeds
    "/feeds/",           # RSS feeds
    "/xmlrpc.php",       # WordPress API
    "/trackback/",       # Trackback URLs
]

# Soft 404 indicators (pages that return 200 but are actually errors)
SOFT_404_INDICATORS = [
    "page not found",
    "404",
    "not found",
    "does not exist",
    "no longer available",
    "has been removed",
    "has been deleted",
    "cannot be found",
    "could not be found",
]

# ============================================================================
# TRAP DETECTION THRESHOLDS - ADJUST BASED ON YOUR DEFINITION OF "LOW VALUE"
# ============================================================================
# The assignment requires you to define "low information value pages"
# Adjust these based on your group's definition:

# Fine-grained: Max URLs per specific pattern (e.g., /pix/action/photo{N}.html)
# Lower = more aggressive filtering of repetitive content
# Higher = more permissive, crawls more similar pages
MAX_FINE_PATTERN_COUNT = 50

# Coarse-grained: Max URLs per path prefix (e.g., /~eppstein/pix/*)
# This prevents traps like photo galleries (2820+ similar pages)
# Set to 0 to disable, or increase if you want more coverage
# Recommended: 100-300 for balance between coverage and trap avoidance
MAX_COARSE_PATTERN_COUNT = 200


def is_trap(url):
    """
    Detect if URL is likely a trap (infinite loop, calendar, etc.)
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        full_path = path + ("?" + query if query else "")
        
        # Check URL length
        if len(url) > MAX_URL_LENGTH:
            return True
        
        # Check path depth
        segments = [s for s in path.split("/") if s]
        if len(segments) > MAX_PATH_DEPTH:
            return True
        
        # Check for repeated path segments (trap indicator)
        if segments:
            segment_counts = defaultdict(int)
            for seg in segments:
                segment_counts[seg] += 1
                if segment_counts[seg] > MAX_REPEATED_SEGMENTS:
                    return True
        
        # Check for trap patterns
        for pattern in COMPILED_TRAP_PATTERNS:
            if pattern.search(full_path):
                return True
        
        # Check for excessive query parameters (often a trap sign)
        if query:
            param_count = query.count("=")
            if param_count > 5:
                return True
        
        return False
        
    except Exception:
        return True


def get_url_pattern(url):
    """
    Extract URL pattern for trap detection.
    Replaces numbers with {N} to identify repeating patterns.
    Returns both fine-grained and coarse patterns for multi-level detection.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path

        # Fine-grained: Replace all numbers and common variations
        fine_pattern = re.sub(r'\d+', '{N}', path)
        fine_pattern = re.sub(r'[a-zA-Z]\d+', 'X{N}', fine_pattern)  # like photo1, img2

        # Coarse-grained: Get path prefix (up to 3 levels) for broader detection
        parts = [p for p in path.split('/') if p]
        if len(parts) > 3:
            coarse_prefix = '/' + '/'.join(parts[:3]) + '/*'
        else:
            coarse_prefix = path

        # Return both patterns
        return f"{parsed.netloc}{fine_pattern}", f"{parsed.netloc}{coarse_prefix}"
    except Exception:
        return url, url


# ============================================================================
# ALLOWED DOMAINS
# ============================================================================

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

COMPILED_DOMAIN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ALLOWED_DOMAIN_PATTERNS]


def is_allowed_domain(hostname):
    """Check if hostname is in allowed domains."""
    hostname = hostname.lower()
    # Remove port if present
    if ":" in hostname:
        hostname = hostname.split(":")[0]
    
    for pattern in COMPILED_DOMAIN_PATTERNS:
        if pattern.match(hostname):
            return True
    return False


# ============================================================================
# MAIN SCRAPER FUNCTIONS
# ============================================================================

# Set to track exact duplicates
exact_hashes = set()
exact_hashes_lock = Lock()


def scraper(url, resp):
    """
    Main scraper function - extracts links and records analytics.
    
    Args:
        url: The URL that was used to get the page
        resp: Response object with status, raw_response, etc.
    
    Returns:
        List of valid URLs to crawl next
    """
    global analytics, exact_hashes
    
    # Normalize and defragment URL
    url = normalize_url(url)
    
    # Check response status
    if resp.status != 200:
        return []

    # Check if we have content
    if not resp.raw_response or not resp.raw_response.content:
        return []

    # Get content
    try:
        content = resp.raw_response.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
    except Exception:
        return []

    # Check for empty or very small content
    if len(content) < 100:
        return []

    # Check for excessively large files (likely not useful text content)
    if len(content) > MAX_CONTENT_SIZE:
        return []

    # Extract links early so we still crawl even if analytics skip the page
    links = extract_links(content, resp.raw_response.url if resp.raw_response else url)

    # Filter and return valid links
    valid_links = []
    invalid_reasons = defaultdict(int) if CRAWLER_DEBUG else None
    invalid_samples = [] if CRAWLER_DEBUG else None

    for link in links:
        ok, reason = classify_url(link)
        if ok:
            valid_links.append(link)
        elif CRAWLER_DEBUG:
            invalid_reasons[reason] += 1
            if len(invalid_samples) < MAX_DEBUG_INVALID:
                invalid_samples.append((link, reason))

    if CRAWLER_DEBUG:
        print(f"[DEBUG] {url} extracted {len(links)} links, "
              f"{len(valid_links)} valid, {len(links) - len(valid_links)} invalid")
        if invalid_reasons:
            summary = ", ".join(
                f"{reason}: {count}" for reason, count in
                sorted(invalid_reasons.items(), key=lambda x: (-x[1], x[0]))
            )
            print(f"[DEBUG] Invalid reasons: {summary}")
        for sample_url, sample_reason in invalid_samples:
            print(f"[DEBUG] Invalid: {sample_reason} -> {sample_url}")

    # Check for exact duplicates (analytics only)
    content_hash = compute_exact_hash(content)
    with exact_hashes_lock:
        if content_hash in exact_hashes:
            return valid_links
        exact_hashes.add(content_hash)

    # Extract text and words
    text, words, word_count = extract_text_and_words(content)

    # Check for soft 404s (pages that return 200 but are error pages)
    if is_soft_404(text):
        return valid_links

    # Check for low information content (analytics only)
    if has_low_information(text, word_count):
        return valid_links

    # Compute SimHash for near-duplicate detection (analytics only)
    simhash = compute_simhash(words)
    if analytics.is_near_duplicate(simhash, threshold=3):
        return valid_links
    analytics.add_simhash(url, simhash)

    # Get subdomain
    subdomain = get_subdomain(url)

    # Record analytics
    analytics.add_page(url, word_count, words, subdomain)

    # Log progress and save reports periodically
    stats = analytics.get_stats()

    # Save every 50 pages for better crash recovery
    if stats["unique_pages"] % 50 == 0:
        analytics.save_report()
        analytics.save_trap_report()

    # Log progress every 100 pages (less verbose)
    if stats["unique_pages"] % 100 == 0:
        print(f"\n[PROGRESS] Unique pages: {stats['unique_pages']}, "
              f"Longest page: {stats['longest_page']['word_count']} words, "
              f"Subdomains: {stats['total_subdomains']}\n")

    return valid_links


def classify_url(url):
    """
    Decide whether to crawl this url or not.
    Returns (True, None) if the URL should be crawled, otherwise (False, reason).
    """
    try:
        # Parse URL
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in {"http", "https"}:
            analytics.record_trap(url, "invalid_scheme")
            return False, "scheme"

        # Check if domain is allowed
        if not is_allowed_domain(parsed.netloc):
            analytics.record_trap(url, "wrong_domain")
            return False, "domain"

        # Check for low-value paths (wp-content, assets, etc.)
        if is_low_value_path(url):
            analytics.record_trap(url, "low_value_path")
            return False, "low_value_path"

        # Check for file extensions we want to skip
        path_lower = parsed.path.lower()

        # Skip common non-HTML file types
        skip_extensions = (
            ".css", ".js", ".bmp", ".gif", ".jpg", ".jpeg", ".ico",
            ".png", ".tiff", ".tif", ".mid", ".mp2", ".mp3", ".mp4",
            ".wav", ".avi", ".mov", ".mpeg", ".ram", ".m4v", ".mkv",
            ".ogg", ".ogv", ".pdf", ".ps", ".eps", ".tex", ".ppt",
            ".pptx", ".doc", ".docx", ".xls", ".xlsx", ".names",
            ".data", ".dat", ".exe", ".bz2", ".tar", ".msi", ".bin",
            ".7z", ".psd", ".dmg", ".iso", ".epub", ".dll", ".cnf",
            ".tgz", ".sha1", ".thmx", ".mso", ".arff", ".rtf", ".jar",
            ".csv", ".rm", ".smil", ".wmv", ".swf", ".wma", ".zip",
            ".rar", ".gz", ".img", ".sql", ".db", ".sqlite", ".json",
            ".xml", ".rss", ".atom", ".apk", ".war", ".ear", ".class",
            ".pyc", ".pyo", ".so", ".o", ".a", ".lib", ".deb", ".rpm",
            ".pkg", ".mpg", ".flv", ".webm", ".svg", ".ttf", ".woff",
            ".woff2", ".eot", ".otf", ".bak", ".tmp", ".log", ".out",
            ".mat", ".m", ".r", ".ipynb", ".nb", ".ss", ".ppsx",
            # Source code files (crawled from Apache directory listings)
            ".cc", ".h", ".hpp", ".cpp", ".c", ".java", ".py", ".pl",
            ".sh", ".scm", ".rkt", ".odc", ".conf", ".dsw", ".dsp",
            ".inc", ".sas", ".fig", ".cls", ".tsv", ".txt"
        )

        if any(path_lower.endswith(ext) for ext in skip_extensions):
            analytics.record_trap(url, "file_extension")
            return False, "extension"

        # Check for traps
        if is_trap(url):
            analytics.record_trap(url, "trap_pattern")
            return False, "trap"

        # Check URL pattern frequency (trap detection) - multi-level
        fine_pattern, coarse_pattern = get_url_pattern(url)

        # Check fine-grained pattern (e.g., /pix/action/photo{N}.html)
        fine_count = analytics.record_url_pattern(fine_pattern)
        if fine_count > MAX_FINE_PATTERN_COUNT:
            analytics.record_trap(url, "pattern_frequency_fine", fine_pattern)
            return False, "pattern_frequency_fine"

        # Check coarse-grained pattern (e.g., /~eppstein/pix/*)
        if MAX_COARSE_PATTERN_COUNT > 0:  # 0 = disabled
            coarse_count = analytics.record_url_pattern(coarse_pattern)
            if coarse_count > MAX_COARSE_PATTERN_COUNT:
                analytics.record_trap(url, "pattern_frequency_coarse", coarse_pattern)
                return False, "pattern_frequency_coarse"

        return True, None

    except TypeError:
        print(f"TypeError for {url}")
        return False, "type_error"
    except Exception as e:
        print(f"Error validating URL {url}: {e}")
        return False, "exception"


def is_valid(url):
    """Return True if the URL should be crawled."""
    ok, _ = classify_url(url)
    return ok


def print_final_report():
    """Print and save the final analytics report."""
    global analytics
    
    print("\n" + "=" * 80)
    print("CRAWLER ANALYTICS REPORT")
    print("=" * 80)
    
    # 1. Unique pages
    stats = analytics.get_stats()
    print(f"\n1. UNIQUE PAGES: {stats['unique_pages']}")
    
    # 2. Longest page
    print(f"\n2. LONGEST PAGE:")
    print(f"   URL: {stats['longest_page']['url']}")
    print(f"   Word count: {stats['longest_page']['word_count']}")
    
    # 3. Top 50 words
    print(f"\n3. TOP 50 MOST COMMON WORDS:")
    top_words = analytics.get_top_words(50)
    for i, (word, count) in enumerate(top_words, 1):
        print(f"   {i:2}. {word}: {count}")
    
    # 4. Subdomains
    subdomain_stats = analytics.get_subdomain_stats()
    print(f"\n4. SUBDOMAINS ({len(subdomain_stats)} total):")
    for subdomain, count in subdomain_stats:
        print(f"   {subdomain}, {count}")
    
    print("\n" + "=" * 80)
    
    # Save to file
    analytics.save_report()
    print("Report saved to crawler_report.json")

    analytics.save_trap_report()
    print("Trap report saved to trap_report.json")


# Register cleanup to print report on exit
import atexit
atexit.register(print_final_report)
