import json
import random
from pathlib import Path

# -----------------------
# Configuration
# -----------------------
CORPUS_BASE = Path.home() / "Documents/Master/Masterarbeit/MA/Code/Scraping"
OUTPUT_CSV = Path.home() / "Documents/Master/Masterarbeit/MA/Code/Data/labels.csv"
CONTEXT_CHARS = 400   # characters around each keyword match to show
MAX_SNIPPETS = 4      # max number of keyword snippets to display

AI_KEYWORDS = [
    " AI ", "artificial intelligence", "llm", "agent", "machine learning",
    "foundation model", "generative", "gpt", "deep learning", "automation",
    "model", "intelligence", "software",
]

# -----------------------
# Load all documents
# -----------------------
def load_all_documents() -> list[dict]:
    docs = []
    for f in sorted(CORPUS_BASE.glob("*/documents*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return docs

# -----------------------
# Load already labelled URLs
# -----------------------
def load_labelled_urls() -> set[str]:
    labelled = set()
    if not OUTPUT_CSV.exists():
        return labelled
    with OUTPUT_CSV.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("url"):
                continue  # skip header
            parts = line.strip().split("\t")
            if parts:
                labelled.add(parts[0])
    return labelled

# -----------------------
# Save a single label
# -----------------------
def save_label(doc: dict, label: int):
    write_header = not OUTPUT_CSV.exists()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("url\tfund\ttitle\tpublished_time\tlabel\n")
        f.write(
            f"{doc['url']}\t{doc.get('fund','')}\t"
            f"{doc.get('title','')}\t{doc.get('published_time','')}\t{label}\n"
        )

# -----------------------
# Extract keyword snippets
# -----------------------
def get_keyword_snippets(text: str, keywords: list[str], context: int, max_snippets: int) -> list[str]:
    """Finds keyword matches in text and returns surrounding context snippets."""
    text_lower = text.lower()
    snippets = []
    seen_positions = []

    for keyword in keywords:
        start = 0
        while True:
            pos = text_lower.find(keyword.lower(), start)
            if pos == -1:
                break

            # Skip if too close to an already shown snippet
            if any(abs(pos - seen) < context for seen in seen_positions):
                start = pos + 1
                continue

            # Extract surrounding context
            snippet_start = max(0, pos - context)
            snippet_end = min(len(text), pos + len(keyword) + context)
            snippet = text[snippet_start:snippet_end].replace("\n", " ").strip()

            # Mark the keyword with >> << for visibility
            keyword_in_snippet = text[pos:pos + len(keyword)]
            snippet = snippet.replace(keyword_in_snippet, f">>{keyword_in_snippet.upper()}<<")

            snippets.append(snippet)
            seen_positions.append(pos)
            start = pos + 1

            if len(snippets) >= max_snippets:
                return snippets

    return snippets

# -----------------------
# Main labelling loop
# -----------------------
def run_labelling(n: int = 400):
    print("Loading documents...")
    docs = load_all_documents()
    labelled_urls = load_labelled_urls()

    unlabelled = [d for d in docs if d.get("url") not in labelled_urls]
    random.shuffle(unlabelled)

    print(f"\n  Total documents:     {len(docs)}")
    print(f"  Already labelled:    {len(labelled_urls)}")
    print(f"  Remaining to label:  {len(unlabelled)}")
    print(f"  Target:              {n}")
    print()
    print("  LABELS:")
    print("  1 = AI-relevant (discusses AI, ML, LLM, automation etc.)")
    print("  0 = Not relevant (pure investment announcement, unrelated topic)")
    print("  s = Skip this document")
    print("  q = Quit and save progress")
    print()

    labelled_this_session = 0

    for doc in unlabelled:
        if labelled_this_session >= n:
            break

        title = doc.get("title", "(no title)")
        fund = doc.get("fund", "")
        date = doc.get("published_time", "")[:10]
        url = doc.get("url", "")
        text = doc.get("text", "")

        # Get keyword snippets
        snippets = get_keyword_snippets(text, AI_KEYWORDS, CONTEXT_CHARS, MAX_SNIPPETS)

        print("\n" + "═" * 65)
        print(f"[{labelled_this_session + 1}/{n}]  {fund}  |  {date}")
        print(f"TITLE:  {title}")
        print(f"URL:    {url}")
        print()

        if snippets:
            print(f"KEYWORD MATCHES ({len(snippets)} shown):")
            for i, snippet in enumerate(snippets, 1):
                print(f"  [{i}] ...{snippet}...")
        else:
            # No keyword match — show beginning of text
            print("NO KEYWORD MATCH — first 400 chars:")
            print(f"  {text[:400].replace(chr(10), ' ')}...")
        print()

        while True:
            raw = input("  Label (1 / 0 / s / q): ").strip().lower()
            if raw == "q":
                print(f"\n  Saved {labelled_this_session} labels. Bye!")
                return
            elif raw == "s":
                break
            elif raw in ("0", "1"):
                save_label(doc, int(raw))
                labelled_this_session += 1
                break
            else:
                print("  Please enter 1, 0, s, or q.")

    print(f"\n  Done! {labelled_this_session} labels saved to {OUTPUT_CSV}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    run_labelling(n=400)
