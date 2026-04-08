import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle

# -----------------------
# Configuration
# -----------------------
LABELS_CSV   = Path.home() / "Documents/Master/Masterarbeit/MA/Code/Data/labels.csv"
CORPUS_DIR   = Path.home() / "Documents/Master/Masterarbeit/MA/Code/Scraping"
OUTPUT_JSONL = Path.home() / "Documents/Master/Masterarbeit/MA/Code/Data/documents_classified.jsonl"
MODEL_FILE   = Path.home() / "Documents/Master/Masterarbeit/MA/Code/Data/classifier.pkl"

# -----------------------
# Load corpus
# -----------------------
def load_corpus(corpus_dir: Path) -> list[dict]:
    docs = []
    for f in sorted(corpus_dir.glob("*/documents*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"  Loaded {len(docs)} documents from corpus")
    return docs

# -----------------------
# Build input text
# -----------------------
def build_input_text(doc: dict) -> str:
    """Combines title and text for classification input.
    Title is repeated to give it more weight.
    """
    title = doc.get("title", "")
    text  = doc.get("text", "")
    return f"{title} {title} {text}"

# -----------------------
# Train classifier
# -----------------------
def train(labels_csv: Path, corpus_dir: Path):
    print("\n=== Loading labels ===")
    labels_df = pd.read_csv(labels_csv, sep="\t")
    print(f"  Labels loaded: {len(labels_df)}")
    print(f"  Class distribution:\n{labels_df['label'].value_counts().to_string()}")
    print(f"  AI-relevant: {labels_df['label'].mean():.1%}")

    print("\n=== Loading corpus ===")
    all_docs = load_corpus(corpus_dir)
    url_to_doc = {d["url"]: d for d in all_docs}

    # Match labels to corpus documents
    labelled_docs = []
    for _, row in labels_df.iterrows():
        url = row["url"]
        if url in url_to_doc:
            doc = url_to_doc[url].copy()
            doc["label"] = int(row["label"])
            labelled_docs.append(doc)

    print(f"  Matched {len(labelled_docs)} labelled documents")

    texts  = [build_input_text(d) for d in labelled_docs]
    labels = [d["label"] for d in labelled_docs]

    # Train/test split (80/20, stratified to preserve class balance)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"\n  Train: {len(train_texts)}  |  Test: {len(test_texts)}")

    print("\n=== Training TF-IDF + Logistic Regression ===")

    # TF-IDF: converts text to numerical features
    # ngram_range=(1,2): uses single words AND word pairs
    # max_features: keeps top 20k most informative features
    # sublinear_tf: dampens very frequent terms
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    # class_weight='balanced' handles imbalanced labels automatically
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
    )
    clf.fit(X_train, train_labels)

    print("\n=== Evaluation on test set ===")
    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    f1  = f1_score(test_labels, preds, average="weighted")

    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print()
    print(classification_report(
        test_labels, preds,
        target_names=["Not relevant", "AI-relevant"]
    ))

    # Cross-validation for more robust estimate
    print("=== Cross-validation (5-fold) ===")
    X_all = vectorizer.transform(texts)
    cv_scores = cross_val_score(clf, X_all, labels, cv=5, scoring="f1_weighted")
    print(f"  F1 scores: {[round(s, 3) for s in cv_scores]}")
    print(f"  Mean F1:   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Save model to disk
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_FILE.open("wb") as f:
        pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)
    print(f"\n  Model saved to {MODEL_FILE}")

    return vectorizer, clf, all_docs

# -----------------------
# Classify full corpus
# -----------------------
def classify_corpus(vectorizer, clf, all_docs: list[dict]):
    print("\n=== Classifying full corpus ===")
    texts = [build_input_text(d) for d in all_docs]
    X = vectorizer.transform(texts)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]  # probability of class 1 (AI-relevant)

    relevant = sum(preds)
    print(f"  AI-relevant:  {relevant} ({relevant/len(preds):.1%})")
    print(f"  Not relevant: {len(preds) - relevant} ({(len(preds)-relevant)/len(preds):.1%})")

    # Save classified corpus
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for doc, pred, prob in zip(all_docs, preds, probs):
            doc["ai_relevant"]        = int(pred)
            doc["ai_relevance_score"] = round(float(prob), 4)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n  Classified corpus saved to {OUTPUT_JSONL}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    vectorizer, clf, all_docs = train(LABELS_CSV, CORPUS_DIR)
    classify_corpus(vectorizer, clf, all_docs)
