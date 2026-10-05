"""Answer a question from the notes, with citations, or refuse.

Two gates protect against making things up:
  1. Retrieval gate (always on, no API): refuse unless the question's words appear in the notes
     (coverage) and the best chunk scores above a minimum.
  2. Generation gate (Claude mode only): the model is told to reply NOT_IN_NOTES if the retrieved
     passages do not contain the answer.

Default mode is extractive: it returns the most relevant sentences from the best chunks, quoted,
so every statement is traceable to the notes. Claude mode writes a fluent answer from the same chunks.
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .index import Chunk, Index, load_chunks, tokenize

MIN_COVERAGE = 0.6   # tuned on the dev split, see study/evaluate.py
MIN_SCORE = 4.0
NOTES = Path(__file__).resolve().parent.parent / "notes"
MODEL = os.environ.get("STUDY_MODEL", "claude-sonnet-5-5")
REFUSAL = "I can't answer that from the study notes."


@dataclass
class Answer:
    answered: bool
    text: str
    citations: list[str] = field(default_factory=list)
    reason: str = ""

    def __str__(self) -> str:
        if not self.answered:
            return f"{self.text} ({self.reason})"
        return f"{self.text}\n\nSources: " + "; ".join(self.citations)


def load_index() -> Index:
    return Index(load_chunks(NOTES))


def gate(index: Index, question: str, min_coverage=MIN_COVERAGE, min_score=MIN_SCORE):
    """Return (hits, refusal_reason). refusal_reason is '' when retrieval looks good enough."""
    q = tokenize(question)
    cov = index.coverage(q)
    hits = index.search(question, k=3)
    if cov < min_coverage:
        return hits, f"only {cov:.0%} of the question's words appear in the notes"
    if hits[0][0] < min_score:
        return hits, f"best match is weak (score {hits[0][0]:.1f})"
    return hits, ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extractive_answer(question: str, hits: list[tuple[float, Chunk]], max_sentences: int = 3) -> Answer:
    q = set(tokenize(question))
    top_score = hits[0][0]
    picked = []
    for score, chunk in hits:
        if score < top_score * 0.6:  # ignore much weaker chunks
            continue
        for s in _sentences(chunk.text):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                picked.append((overlap, score, s, chunk))
    picked.sort(key=lambda x: (-x[0], -x[1]))
    chosen = picked[:max_sentences]
    if not chosen:
        return Answer(False, REFUSAL, reason="no sentence in the notes matches the question")
    cites = list(dict.fromkeys(c.cite for _, _, _, c in chosen))
    return Answer(True, " ".join(f'"{s}"' for _, _, s, _ in chosen), cites)


def claude_answer(question: str, hits, client=None) -> Answer:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    sources = "\n\n".join(f"[{i}] ({c.cite}) {c.text}" for i, (_, c) in enumerate(hits, 1))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=("You are a study assistant. Answer ONLY from the numbered sources. Cite sources like [1]. "
                "If the sources do not contain the answer, reply with exactly NOT_IN_NOTES. "
                "Never use outside knowledge."),
        messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text == "NOT_IN_NOTES":
        return Answer(False, REFUSAL, reason="the model found no answer in the retrieved notes")
    used = [int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)]
    cites = list(dict.fromkeys(hits[n - 1][1].cite for n in used))
    if not cites:  # an answer with no citations is not acceptable
        return Answer(False, REFUSAL, reason="the model's answer cited no source")
    return Answer(True, text, cites)


def answer(question: str, index: Index | None = None, use_claude: bool = False, client=None,
           min_coverage=MIN_COVERAGE, min_score=MIN_SCORE) -> Answer:
    index = index or load_index()
    hits, why = gate(index, question, min_coverage, min_score)
    if why:
        return Answer(False, REFUSAL, reason=why)
    return claude_answer(question, hits, client) if use_claude else extractive_answer(question, hits)
