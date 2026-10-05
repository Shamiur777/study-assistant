"""Evaluate retrieval + refusal on a labelled question set, with a held-out test split.

Questions alternate into 'dev' and 'test'. Thresholds are tuned on dev only, then reported once on test,
so the test numbers are not inflated by tuning.

  python -m study.evaluate
"""
import itertools
import json
from pathlib import Path

from .answer import MIN_COVERAGE, MIN_SCORE, answer, load_index

QUESTIONS = Path(__file__).resolve().parent.parent / "data" / "questions.json"


def split():
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    ans = [q for q in qs if q["doc"]]
    una = [q for q in qs if not q["doc"]]
    dev = ans[0::2] + una[0::2]
    test = ans[1::2] + una[1::2]
    return dev, test


def score(questions, index, cov, sc) -> dict:
    r = dict(n=len(questions), answer_correct=0, answer_wrong_source=0, wrongly_refused=0,
             refused_ok=0, hallucinated=0, details=[])
    for q in questions:
        a = answer(q["q"], index, min_coverage=cov, min_score=sc)
        if q["doc"]:
            if not a.answered:
                r["wrongly_refused"] += 1
                r["details"].append(("WRONGLY REFUSED", q["q"], a.reason))
            elif any(c.startswith(q["doc"] + " >") for c in a.citations):
                r["answer_correct"] += 1
            else:
                r["answer_wrong_source"] += 1
                r["details"].append(("WRONG SOURCE", q["q"], ", ".join(a.citations)))
        else:
            if a.answered:
                r["hallucinated"] += 1
                r["details"].append(("SHOULD HAVE REFUSED", q["q"], ", ".join(a.citations)))
            else:
                r["refused_ok"] += 1
    r["accuracy"] = (r["answer_correct"] + r["refused_ok"]) / r["n"]
    return r


def show(title, r):
    n_ans = r["answer_correct"] + r["answer_wrong_source"] + r["wrongly_refused"]
    n_una = r["refused_ok"] + r["hallucinated"]
    print(f"{title}: accuracy {r['accuracy']:.0%} ({r['n']} questions)")
    print(f"  answerable ({n_ans}): {r['answer_correct']} correct, {r['answer_wrong_source']} wrong source, "
          f"{r['wrongly_refused']} wrongly refused")
    print(f"  unanswerable ({n_una}): {r['refused_ok']} refused, {r['hallucinated']} answered anyway")
    for kind, q, why in r["details"]:
        print(f"    {kind}: {q}  [{why}]")


def main():
    index = load_index()
    dev, test = split()
    grid = itertools.product([0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0], [1, 2, 3, 4, 5, 6, 8])
    results = []
    for cov, sc in grid:
        r = score(dev, index, cov, sc)
        # best dev accuracy; tie-break toward fewer hallucinations (wrong answers cost more than refusals)
        results.append(((r["accuracy"], -r["hallucinated"]), cov, sc))
    _, cov, sc = max(results)
    print(f"Tuned on dev set: min_coverage={cov}, min_score={sc}  (current defaults: {MIN_COVERAGE}, {MIN_SCORE})\n")
    show("DEV (tuned)", score(dev, index, cov, sc))
    print()
    show("TEST (held out)", score(test, index, cov, sc))


if __name__ == "__main__":
    main()
