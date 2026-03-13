import json
import glob
from pathlib import Path
from collections import defaultdict

def mean(lst): return sum(lst) / len(lst) if lst else 0
def median(lst):
    s = sorted(lst)
    n = len(s)
    return (s[n//2] + s[~n//2]) / 2 if n else 0
def stdev(lst):
    if len(lst) < 2: return 0
    m = mean(lst)
    return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

BASE = Path.home() / "Documents/Master/Masterarbeit/MA/Code"
SCRAPING = BASE / "Scraping"
CLASSIFIED = BASE / "Data/documents_classified.jsonl"

docs = []
files = glob.glob(str(SCRAPING / "**/documents*.jsonl"), recursive=True)
print(f"Found {len(files)} JSONL file(s)")
for filepath in files:
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try: docs.append(json.loads(line))
                except: pass

print(f"Total documents loaded: {len(docs)}")
if docs: print(f"Field names: {list(docs[0].keys())}\n")

def word_count(doc):
    text = doc.get("text") or doc.get("content") or doc.get("body") or ""
    return len(text.split())

def get_fund(doc):
    return doc.get("fund") or doc.get("source") or doc.get("fund_name") or "Unknown"

def get_year(doc):
    date = doc.get("published_time") or doc.get("date") or doc.get("published") or ""
    if date and len(str(date)) >= 4:
        try:
            y = int(str(date)[:4])
            if 2000 <= y <= 2030: return y
        except: pass
    return None

fund_counts = defaultdict(int)
fund_words = defaultdict(list)
for doc in docs:
    fund = get_fund(doc)
    wc = word_count(doc)
    fund_counts[fund] += 1
    fund_words[fund].append(wc)

print("=" * 60)
print("DOCUMENTS PER FUND")
print("=" * 60)
print(f"{'Fund':<30} {'Docs':>6} {'Avg words':>10}")
print("-" * 60)
for fund, count in sorted(fund_counts.items(), key=lambda x: -x[1]):
    avg = round(mean(fund_words[fund]))
    print(f"{fund:<30} {count:>6} {avg:>10}")
print(f"{'TOTAL':<30} {sum(fund_counts.values()):>6}")

year_counts = defaultdict(int)
for doc in docs:
    y = get_year(doc)
    if y: year_counts[y] += 1

print("\n" + "=" * 60)
print("TEMPORAL DISTRIBUTION")
print("=" * 60)
for year in sorted(year_counts):
    bar = "█" * (year_counts[year] // 10)
    print(f"{year}: {year_counts[year]:>5}  {bar}")

all_words = [word_count(d) for d in docs]
nz = [w for w in all_words if w > 0]
print("\n" + "=" * 60)
print("DOCUMENT LENGTH (words)")
print("=" * 60)
print(f"  Total words   : {sum(all_words):,}")
print(f"  Mean          : {round(mean(nz)):,}")
print(f"  Median        : {round(median(nz)):,}")
print(f"  Min           : {min(nz):,}")
print(f"  Max           : {max(nz):,}")
print(f"  Stdev         : {round(stdev(nz)):,}")
print(f"  Empty docs    : {all_words.count(0)}")

if CLASSIFIED.exists():
    classified = []
    with open(CLASSIFIED, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try: classified.append(json.loads(line))
                except: pass
    ai = sum(1 for d in classified if d.get("ai_relevant") == 1 or d.get("predicted_label") == 1)
    print("\n" + "=" * 60)
    print("AI RELEVANCE")
    print("=" * 60)
    print(f"  Total         : {len(classified):,}")
    print(f"  AI-relevant   : {ai:,} ({ai/len(classified)*100:.1f}%)")
    print(f"  Not relevant  : {len(classified)-ai:,} ({(len(classified)-ai)/len(classified)*100:.1f}%)")
    fund_ai = defaultdict(lambda: {"total": 0, "relevant": 0})
    for doc in classified:
        f = get_fund(doc)
        fund_ai[f]["total"] += 1
        if doc.get("ai_relevant") == 1 or doc.get("predicted_label") == 1:
            fund_ai[f]["relevant"] += 1
    print(f"\n{'Fund':<30} {'Total':>6} {'AI-rel':>8} {'%':>6}")
    print("-" * 55)
    for f, v in sorted(fund_ai.items(), key=lambda x: -x[1]["total"]):
        pct = v["relevant"]/v["total"]*100 if v["total"] > 0 else 0
        print(f"{f:<30} {v['total']:>6} {v['relevant']:>8} {pct:>5.1f}%")
else:
    print(f"\n[!] Classified file not found at {CLASSIFIED}")

print("\nDone.")
