import json
from pathlib import Path
from collections import Counter, defaultdict

# -----------------------
# Configuration
# -----------------------
CORPUS_BASE = Path.home() / "Documents/Master/Masterarbeit/MA/Code/Scraping"

# Dates that are clearly placeholders and should be removed
SUSPICIOUS_DATES = ["2026-01-01", "2000-01-01", "1970-01-01"]

# -----------------------
# Load all documents
# -----------------------
def load_all_documents() -> list[dict]:
    docs = []
    files = sorted(CORPUS_BASE.glob("*/documents*.jsonl"))
    print(f"Found {len(files)} JSONL files:\n")
    for f in files:
        count = 0
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"  [JSON Error] {f.name}: {e}")
        print(f"  {count:>5}  {f}")
    return docs

# -----------------------
# Validation checks
# -----------------------
def validate(docs: list[dict]):
    print(f"\n{'='*60}")
    print(f"  CORPUS VALIDATION REPORT")
    print(f"{'='*60}\n")

    # ── 1. Total count ──────────────────────────────────────────
    print(f"[1] TOTAL DOCUMENTS: {len(docs)}\n")

    # ── 2. Documents per fund ───────────────────────────────────
    print("[2] DOCUMENTS PER FUND:")
    fund_counts = Counter(d.get("fund", "UNKNOWN") for d in docs)
    for fund, count in sorted(fund_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>5}  {fund}")
    print()

    # ── 3. Missing fields ───────────────────────────────────────
    print("[3] MISSING FIELDS:")
    fields = ["title", "published_time", "author", "text", "url"]
    for field in fields:
        missing = sum(1 for d in docs if not d.get(field, "").strip())
        pct = missing / len(docs) * 100
        print(f"  {field:<20} missing: {missing:>5}  ({pct:.1f}%)")
    print()

    # ── 4. Year distribution ────────────────────────────────────
    print("[4] YEAR DISTRIBUTION:")
    year_counts = Counter()
    no_date = 0
    bad_date = 0
    for d in docs:
        date = d.get("published_time", "")
        if not date or not date.strip():
            no_date += 1
        elif len(date) >= 4 and date[:4].isdigit():
            year_counts[date[:4]] += 1
        else:
            bad_date += 1
    for year in sorted(year_counts):
        bar = "█" * (year_counts[year] // 10)
        print(f"  {year}: {year_counts[year]:>5}  {bar}")
    print(f"  No date:   {no_date:>5}")
    print(f"  Bad date:  {bad_date:>5}")
    print()

    # ── 5. Text length distribution ─────────────────────────────
    print("[5] TEXT LENGTH DISTRIBUTION:")
    char_counts = [d.get("char_count", len(d.get("text", ""))) for d in docs]
    buckets = [
        ("< 500 chars (should be 0)", lambda x: x < 500),
        ("500 – 1k chars",            lambda x: 500 <= x < 1000),
        ("1k – 3k chars",             lambda x: 1000 <= x < 3000),
        ("3k – 10k chars",            lambda x: 3000 <= x < 10000),
        ("> 10k chars",               lambda x: x >= 10000),
    ]
    for label, fn in buckets:
        count = sum(1 for c in char_counts if fn(c))
        print(f"  {label:<30} {count:>5}")
    print(f"  Avg length: {sum(char_counts) // len(char_counts):>5} chars")
    print()

    # ── 6. Duplicate URLs ───────────────────────────────────────
    print("[6] DUPLICATE URLs:")
    url_counts = Counter(d.get("url", "") for d in docs)
    dupes = {url: count for url, count in url_counts.items() if count > 1}
    if dupes:
        print(f"  Found {len(dupes)} duplicate URLs:")
        for url, count in sorted(dupes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count}x  {url}")
    else:
        print("  ✓ No duplicates found")
    print()

    # ── 7. Suspicious dates ─────────────────────────────────────
    print("[7] SUSPICIOUS DATES (possible placeholders):")
    for date in SUSPICIOUS_DATES:
        count = sum(1 for d in docs if d.get("published_time", "").startswith(date))
        if count > 0:
            print(f"  {date}: {count} documents")
    total_suspicious = sum(
        1 for d in docs
        if any(d.get("published_time", "").startswith(s) for s in SUSPICIOUS_DATES)
    )
    pct = total_suspicious / len(docs) * 100
    print(f"  Total suspicious dates: {total_suspicious} ({pct:.1f}% of corpus)")
    print()

    # ── 8. Per-fund date coverage ────────────────────────────────
    print("[8] DATE COVERAGE PER FUND:")
    fund_no_date = defaultdict(int)
    fund_total = defaultdict(int)
    fund_suspicious = defaultdict(int)
    for d in docs:
        fund = d.get("fund", "UNKNOWN")
        fund_total[fund] += 1
        date = d.get("published_time", "")
        if not date or not date.strip():
            fund_no_date[fund] += 1
        if any(date.startswith(s) for s in SUSPICIOUS_DATES):
            fund_suspicious[fund] += 1
    for fund in sorted(fund_total):
        total = fund_total[fund]
        no_d = fund_no_date[fund]
        susp = fund_suspicious[fund]
        print(f"  {fund:<40} total: {total:>4}  no date: {no_d:>4}  suspicious: {susp:>4}")
    print()

    print(f"{'='*60}")
    print("  DONE")
    print(f"{'='*60}\n")

# -----------------------
# Clean corpus
# -----------------------
def clean_corpus():
    """Removes duplicate URLs and documents with missing or placeholder dates.
    Overwrites each JSONL file in place with the cleaned version.
    """
    print(f"\n{'='*60}")
    print(f"  CLEANING CORPUS")
    print(f"{'='*60}\n")

    files = sorted(CORPUS_BASE.glob("*/documents*.jsonl"))
    total_before = 0
    total_after = 0
    total_removed_dupes = 0
    total_removed_dates = 0
    seen_urls = set()  # track duplicates across all files

    for f in files:
        docs = []
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        cleaned = []
        removed_dupes = 0
        removed_dates = 0

        for doc in docs:
            url = doc.get("url", "")
            date = doc.get("published_time", "")

            # Remove duplicates (across all files)
            if url in seen_urls:
                removed_dupes += 1
                total_removed_dupes += 1
                continue
            seen_urls.add(url)

            # Remove documents without valid date or with placeholder dates
            if (not date
                    or not date.strip()
                    or any(date.startswith(s) for s in SUSPICIOUS_DATES)):
                removed_dates += 1
                total_removed_dates += 1
                continue

            cleaned.append(doc)

        # Overwrite file with cleaned version
        with f.open("w", encoding="utf-8") as fh:
            for doc in cleaned:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

        total_before += len(docs)
        total_after += len(cleaned)
        print(f"  {f.parent.name:<30} {len(docs):>4} → {len(cleaned):>4}  "
              f"(-{removed_dupes} dupes, -{removed_dates} bad dates)")

    print(f"\n  Total before: {total_before}")
    print(f"  Total after:  {total_after}")
    print(f"  Removed duplicates:  {total_removed_dupes}")
    print(f"  Removed bad dates:   {total_removed_dates}")
    print(f"\n{'='*60}\n")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    docs = load_all_documents()
    validate(docs)

    answer = input("Clean corpus? Removes duplicates + bad dates (y/n): ").strip().lower()
    if answer == "y":
        clean_corpus()
        print("Re-validating after cleaning...\n")
        docs = load_all_documents()
        validate(docs)
    else:
        print("Cleaning skipped.")
