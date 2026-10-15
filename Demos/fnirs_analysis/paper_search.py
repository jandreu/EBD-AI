"""
paper_search.py

Keyword (BM25) search over the passages in data/papers.json, the open-access
n-back and fNIRS papers collected by build_paper_corpus.py. rag_core.py puts
the best passages into its prompts as background and lists their papers as
sources.

BM25 instead of embeddings keeps the search transparent and dependency-free,
in the same spirit as retrieve_rules(). Spellings are normalised first, so a
British "haemoglobin" question finds American "hemoglobin" text and "2-back",
"2 back" and "2back" are the same word.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

PAPERS_PATH = Path(__file__).resolve().parent / "data" / "papers.json"

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just me more most my no nor not now of off on
once only or other our out over own same she should so some such than that the their them then there these
they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours tell explain mean means meaning please show give does did
""".split())

# Spelling variants mapped to one form before tokenising.
_NORMALISE = [
    (re.compile(r"haem"), "hem"),
    # Deoxy first: "oxygenated hemoglobin" is inside "deoxygenated hemoglobin".
    (re.compile(r"deoxyhemoglobin|deoxygenated hemoglobin|deoxy-hemoglobin|\bdeoxy-hb\b|\bhhb\b"), "hbr"),
    (re.compile(r"\boxyhemoglobin|\boxygenated hemoglobin|\boxy-hemoglobin|\boxy-hb\b|\bhbo2\b"), "hbo"),
    (re.compile(r"\b(\d|n)\s*-?\s*back\b"), r"\1back"),
    (re.compile(r"near-infrared|near infrared"), "nearinfrared"),
    (re.compile(r"working-memory"), "working memory"),
    (re.compile(r"\bdlpfc\b"), "dorsolateral prefrontal"),
    (re.compile(r"\bpfc\b"), "prefrontal"),
]


def normalise(text: str) -> str:
    text = text.lower()
    for pattern, replacement in _NORMALISE:
        text = pattern.sub(replacement, text)
    return text


# Everyday words in questions, and the words papers use for the same idea.
SYNONYMS = {
    "hard": ["load", "workload", "demand"], "harder": ["load", "workload", "demand"],
    "difficult": ["load", "workload", "demand"], "difficulty": ["load", "workload"],
    "easy": ["load", "0back"], "brain": ["cortex", "cortical"], "blood": ["hemodynamic", "oxygenation"],
    "oxygen": ["oxygenation", "hbo"], "memory": ["working"], "sensor": ["channel", "optode"],
    "fall": ["decrease"], "falls": ["decrease"], "drop": ["decrease"], "drops": ["decrease"],
    "rise": ["increase"], "rises": ["increase"], "high": ["increase"], "low": ["decrease"],
}


def query_tokens(text: str) -> list:
    """Tokens of a question plus the synonyms papers use, without repeats."""
    words = tokens(text)
    for word in re.findall(r"[a-z0-9]+", normalise(text)):
        words += SYNONYMS.get(word, [])
    return list(dict.fromkeys(words))


def tokens(text: str) -> list:
    words = re.findall(r"[a-z0-9]+", normalise(text))
    # Light stemming: plural and common suffixes, so "loads"/"loaded" match "load".
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        for suffix in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out


class PaperIndex:
    """BM25 index over every passage of every paper."""

    def __init__(self, papers: list, k1: float = 1.4, b: float = 0.75):
        self.papers = papers
        self.entries = []  # (paper, passage, term counts, length)
        for paper in papers:
            for passage in paper.get("passages", []):
                # The paper title is part of each passage's text for scoring.
                counts = Counter(tokens(passage["text"] + " " + paper.get("citation", "")))
                self.entries.append((paper, passage, counts, sum(counts.values())))
        self.k1, self.b = k1, b
        n = len(self.entries)
        self.avg_len = sum(e[3] for e in self.entries) / n if n else 1.0
        df = Counter()
        for _, _, counts, _ in self.entries:
            df.update(counts.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query_terms: list, counts: Counter, length: int) -> float:
        total = 0.0
        for term in query_terms:
            f = counts.get(term, 0)
            if f:
                norm = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / norm
        return total

    def search(self, query: str, top_k: int = 4, per_paper: int = 2, min_score: float = 2.0) -> list:
        """Best passages for `query`: at most `per_paper` from one paper, none
        below `min_score` (an off-topic question gets nothing rather than
        noise). Each hit is a dict with the paper's reference and the passage."""
        terms = query_tokens(query)
        if not terms:
            return []
        ranked = sorted(
            ((self.score(terms, counts, length), paper, passage)
             for paper, passage, counts, length in self.entries),
            key=lambda item: item[0], reverse=True)
        hits, used = [], Counter()
        for score, paper, passage in ranked:
            if score < min_score or len(hits) >= top_k:
                break
            if used[paper["id"]] >= per_paper:
                continue
            used[paper["id"]] += 1
            hits.append({
                "paper_id": paper["id"], "citation": paper["citation"], "short": paper["short"],
                "url": paper["url"], "licence": paper.get("licence", ""), "section": passage["section"],
                "text": passage["text"], "score": round(score, 2),
            })
        return hits


def load_index(path: Path = PAPERS_PATH):
    """The index over data/papers.json, or None when the file is missing."""
    try:
        papers = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return PaperIndex(papers)


def sources_from_hits(hits: list) -> list:
    """One citation entry per paper, in the shape the API's `citations` use."""
    seen, out = set(), []
    for hit in hits:
        if hit["paper_id"] in seen:
            continue
        seen.add(hit["paper_id"])
        out.append({"citation": hit["citation"], "url": hit["url"], "summary": "",
                    "vetted": True, "source": "papers"})
    return out


def format_excerpts(hits: list) -> str:
    """Passages as tagged, attributed excerpts for a prompt."""
    return "\n".join(
        f'<excerpt cite="{hit["short"]}" section="{hit["section"]}">\n{hit["text"]}\n</excerpt>'
        for hit in hits)
