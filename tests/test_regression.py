"""Guard against quality regressions: the held-out split must stay at or above the published figure."""
from study.answer import MIN_COVERAGE, MIN_SCORE, load_index
from study.evaluate import score, split


def test_held_out_accuracy_does_not_regress():
    _, test = split()
    r = score(test, load_index(), MIN_COVERAGE, MIN_SCORE)
    assert r["accuracy"] >= 0.85, r["details"]


def test_every_answerable_question_expects_a_real_note():
    from study.answer import NOTES
    docs = {p.stem for p in NOTES.glob("*.md")}
    dev, test = split()
    assert all(q["doc"] in docs for q in dev + test if q["doc"])
