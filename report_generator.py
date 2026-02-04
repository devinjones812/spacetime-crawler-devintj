#!/usr/bin/env python3
"""
Report Generator for Web Crawler Analytics.

This script generates a comprehensive report from the crawler's analytics data.
It can be run standalone after crawling to generate the final report.

Usage:
    python report_generator.py

The report includes:
1. Number of unique pages crawled
2. Longest page (by word count)
3. Top 50 most common words (excluding stopwords)
4. Subdomain statistics for uci.edu
"""

import json
import os
from datetime import datetime


def load_report_data(filepath="crawler_report.json"):
    """Load the analytics report from JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Report file '{filepath}' not found.")
        print("Make sure you have run the crawler first.")
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_text_report(data, output_file="REPORT.md"):
    """Generate a readable markdown report from analytics data."""
    
    report_lines = []
    
    # Header
    report_lines.append("# Web Crawler Analytics Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("This report contains analytics from crawling the following UCI domains:")
    report_lines.append("- `*.ics.uci.edu`")
    report_lines.append("- `*.cs.uci.edu`")
    report_lines.append("- `*.informatics.uci.edu`")
    report_lines.append("- `*.stat.uci.edu`")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Question 1: Unique Pages
    report_lines.append("## 1. Unique Pages")
    report_lines.append("")
    unique_count = data.get("unique_pages_count", 0)
    report_lines.append(f"**Total unique pages found:** {unique_count:,}")
    report_lines.append("")
    report_lines.append("*Note: Uniqueness is determined by URL only (excluding fragments). "
                       "For example, `http://www.ics.uci.edu#aaa` and `http://www.ics.uci.edu#bbb` "
                       "are considered the same page.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Question 2: Longest Page
    report_lines.append("## 2. Longest Page")
    report_lines.append("")
    longest = data.get("longest_page", {})
    report_lines.append(f"**URL:** `{longest.get('url', 'N/A')}`")
    report_lines.append("")
    report_lines.append(f"**Word count:** {longest.get('word_count', 0):,} words")
    report_lines.append("")
    report_lines.append("*Note: HTML markup is not counted as words. Only visible text content is counted.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Question 3: Top 50 Words
    report_lines.append("## 3. Top 50 Most Common Words")
    report_lines.append("")
    report_lines.append("*English stop words have been excluded from this list.*")
    report_lines.append("")
    report_lines.append("| Rank | Word | Frequency |")
    report_lines.append("|------|------|-----------|")
    
    top_words = data.get("top_50_words", [])
    for i, (word, count) in enumerate(top_words, 1):
        report_lines.append(f"| {i} | {word} | {count:,} |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Question 4: Subdomains
    report_lines.append("## 4. Subdomains in uci.edu")
    report_lines.append("")
    subdomains = data.get("subdomains", [])
    report_lines.append(f"**Total subdomains found:** {len(subdomains)}")
    report_lines.append("")
    report_lines.append("| Subdomain | Unique Pages |")
    report_lines.append("|-----------|--------------|")
    
    for subdomain, count in subdomains:
        report_lines.append(f"| {subdomain} | {count:,} |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Technical Details
    report_lines.append("## Technical Implementation")
    report_lines.append("")
    report_lines.append("### Crawler Features")
    report_lines.append("")
    report_lines.append("1. **HTML Parsing:** BeautifulSoup with lxml parser")
    report_lines.append("2. **URL Normalization:** Defragmentation, trailing slash removal, "
                       "lowercase schemes/hosts")
    report_lines.append("3. **Duplicate Detection:**")
    report_lines.append("   - Exact: SHA-256 content hashing")
    report_lines.append("   - Near-duplicate: SimHash fingerprinting (64-bit, Hamming distance ≤ 3)")
    report_lines.append("4. **Trap Detection:**")
    report_lines.append("   - Calendar/date URL patterns")
    report_lines.append("   - Infinite pagination")
    report_lines.append("   - Session/action parameters")
    report_lines.append("   - Repeated path segments")
    report_lines.append("   - Excessive URL length (>300 chars)")
    report_lines.append("5. **Low-Information Filtering:**")
    report_lines.append("   - Minimum word threshold (50 words)")
    report_lines.append("   - Alpha-to-content ratio check")
    report_lines.append("6. **Multithreading:** 4 threads with per-domain politeness (500ms delay)")
    report_lines.append("")
    report_lines.append("### Extra Credit Implementation")
    report_lines.append("")
    report_lines.append("- **+2 points:** SimHash similarity detection implemented from scratch")
    report_lines.append("- **+5 points:** Multithreaded crawler with thread-safe frontier and "
                       "per-domain politeness enforcement")
    report_lines.append("")
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated: {output_file}")
    return "\n".join(report_lines)


def print_console_report(data):
    """Print a summary report to console."""
    print("\n" + "=" * 80)
    print("CRAWLER ANALYTICS REPORT")
    print("=" * 80)
    
    # 1. Unique pages
    print(f"\n1. UNIQUE PAGES: {data.get('unique_pages_count', 0):,}")
    
    # 2. Longest page
    longest = data.get("longest_page", {})
    print(f"\n2. LONGEST PAGE:")
    print(f"   URL: {longest.get('url', 'N/A')}")
    print(f"   Word count: {longest.get('word_count', 0):,}")
    
    # 3. Top 50 words
    print(f"\n3. TOP 50 MOST COMMON WORDS:")
    top_words = data.get("top_50_words", [])
    for i, (word, count) in enumerate(top_words, 1):
        print(f"   {i:2}. {word}: {count:,}")
    
    # 4. Subdomains
    subdomains = data.get("subdomains", [])
    print(f"\n4. SUBDOMAINS ({len(subdomains)} total):")
    for subdomain, count in subdomains:
        print(f"   {subdomain}, {count}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point for report generation."""
    print("Loading crawler analytics data...")
    
    data = load_report_data()
    if data is None:
        return 1
    
    # Print to console
    print_console_report(data)
    
    # Generate markdown report
    generate_text_report(data)
    
    print("\nReport generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
