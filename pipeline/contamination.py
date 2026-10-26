"""Text-overlap detection for the Phase 2 leakage scan (safety invariant #4 /
Work Plan §10). Pure functions — fully implemented and unit-tested; they run as
soon as the corpora exist, no models required.

Primary signal: n-gram *containment* — the fraction of a probe item's n-grams
that also appear anywhere in a corpus. High containment of a benchmark item in
the capability/fine-tuning corpus means the utility probe may be contaminated.
Secondary signal: exact (whitespace-normalized) duplicates.
"""

import re

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def shingles(text: str, n: int) -> set[str]:
    """Set of n-gram token shingles. Short texts (< n tokens) collapse to one
    shingle of their whole token sequence so exact short items can still match."""
    toks = tokenize(text)
    if not toks:
        return set()
    if len(toks) < n:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def corpus_shingles(texts: list[str], n: int) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out |= shingles(t, n)
    return out


def containment(text: str, corpus_shingle_set: set[str], n: int) -> float:
    sh = shingles(text, n)
    if not sh:
        return 0.0
    return len(sh & corpus_shingle_set) / len(sh)


def exact_duplicates(probe_items: list[str], corpus_texts: list[str]) -> list[int]:
    norm = lambda s: " ".join(tokenize(s))
    corpus = {norm(t) for t in corpus_texts}
    return [i for i, item in enumerate(probe_items) if norm(item) in corpus]


def scan_leakage(probe_items: list[str], corpus_texts: list[str],
                 n: int, threshold: float) -> dict:
    """Flag probe items whose n-gram containment in the corpus >= threshold,
    plus any exact duplicates."""
    cs = corpus_shingles(corpus_texts, n)
    dupes = set(exact_duplicates(probe_items, corpus_texts))
    flagged = []
    for i, item in enumerate(probe_items):
        score = containment(item, cs, n)
        if score >= threshold or i in dupes:
            flagged.append({"index": i, "containment": round(score, 4),
                            "exact_duplicate": i in dupes,
                            "preview": item[:80]})
    return {
        "n": n, "threshold": threshold,
        "n_probe": len(probe_items), "n_corpus": len(corpus_texts),
        "n_flagged": len(flagged), "clean": not flagged, "flagged": flagged,
    }
