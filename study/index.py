"""Load the notes, split them into section chunks and rank chunks for a question with BM25.

No external dependencies. Each chunk is one '## Section' of one note, so a citation points at a
specific place a student can go and read: 'heart > Blood flow'.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set("""a an the of in on at to for from by with and or but is are was were be been being do does did
what which who whom whose how why when where that this these those it its as into than then there their they them
can could would should will shall may might about between during after before over under up down out off not no
if so such also each other some any all both more most one two give list explain describe tell me us you your
name define briefly main role function functions called called work works made make""".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


@dataclass
class Chunk:
    doc: str
    section: str
    text: str

    @property
    def cite(self) -> str:
        return f"{self.doc} > {self.section}"


def load_chunks(notes_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(notes_dir.glob("*.md")):
        section, body = None, []
        for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:
            if line.startswith("## "):
                if section and body:
                    chunks.append(Chunk(path.stem, section, " ".join(body).strip()))
                section, body = line[3:].strip(), []
            elif line.strip() and not line.startswith("# "):
                body.append(line.strip())
    return chunks


class Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # Index the section title too, so 'blood flow' matches the 'Blood flow' section.
        self.docs = [tokenize(f"{c.section} {c.text}") for c in chunks]
        self.vocab = {t for d in self.docs for t in d}
        self.avg = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def coverage(self, query_tokens: list[str]) -> float:
        """Share of the question's content words that appear anywhere in the notes."""
        if not query_tokens:
            return 0.0
        return sum(t in self.vocab for t in query_tokens) / len(query_tokens)

    def search(self, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
        q = tokenize(query)
        scored = []
        for chunk, doc in zip(self.chunks, self.docs):
            tf = Counter(doc)
            score = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avg))
            scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]
