import json
from pathlib import Path

# Remove known duplicate files
DUPLICATES_TO_REMOVE = [
    Path.home() / "Documents/Master/Masterarbeit/MA/Code/Scraping/aifund_corpus/documents.jsonl",
]

for f in DUPLICATES_TO_REMOVE:
    if f.exists():
        f.unlink()
        print(f"  ✓ Deleted: {f}")
    else:
        print(f"  Already gone: {f}")
