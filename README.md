# Study Assistant (cited answers, honest refusals)

A question-answering assistant over a small set of human anatomy notes. It answers only from the notes, shows exactly which section each answer came from, and says "I can't answer that" when the notes don't cover the question.

The notes are original and the project uses no external data or API to run.

## The problem
AI study tools often answer confidently from memory, with no way for a student to check the source. For learning, a wrong answer with no citation is worse than no answer. This assistant is built around two rules: every answer is traceable, and a missing answer is stated plainly.

## How it works

```
question -> BM25 search over note sections -> retrieval gate -> answer + citations
                                                     |
                                                     +-> refuse (with the reason)
```

- `study/index.py`: splits each note into `## section` chunks (30 chunks across 10 notes) and ranks them with BM25, written from scratch. A citation reads like `heart > Blood flow`.
- `study/answer.py`:
  - **Retrieval gate:** refuse unless enough of the question's words appear in the notes (coverage) and the best chunk scores above a minimum.
  - **Default mode (extractive):** returns the most relevant sentences from the best chunks, quoted, so nothing is invented. Needs no API.
  - **Claude mode (`--claude`):** Claude writes a fluent answer from the same retrieved chunks. It must cite sources like `[1]`, must reply `NOT_IN_NOTES` if the chunks don't answer the question, and an answer with no citation is rejected. This is a second gate behind the retrieval gate.
- `study/evaluate.py`: scores answers and refusals on a labelled question set.

## Evaluation
48 questions: 30 answerable (each with the note it should cite) and 18 unanswerable. The unanswerable ones include out-of-domain questions ("capital of France") and harder near-domain ones ("structure of the eye", "how does an insulin pump work?").

Questions are split alternately into a **dev** set and a **held-out test** set. The two thresholds (minimum coverage and minimum score) were tuned on dev only, then measured once on test.

| Split | Overall | Answerable: correct | Unanswerable: correctly refused |
|---|---|---|---|
| Dev (tuned on) | 100% (25 q) | 15 of 15 | 10 of 10 |
| Test (held out) | 88% (24 q) | 14 of 15 | 7 of 9 |

Test failures, shown as they are:
- "What does bile do?" was wrongly refused. The relevant chunk scored just under the threshold.
- "How does the liver break down alcohol?" and "How does a frog heart differ from a human heart?" were answered from loosely related chunks. Their words exist in the notes, so the retrieval gate can't tell the question is out of scope.

The last two failures are the reason Claude mode has its own gate: a model can read the retrieved text and recognise that it doesn't answer the question. I haven't run Claude mode against the real API yet, so I make no accuracy claim for it.

Caveat: 48 questions is small. These numbers show the method works and where it breaks, not how it would perform on a full syllabus.

## Run it

```bash
python -m pytest -q                      # 9 tests, no API key needed
python -m study.ask "What does the left ventricle do?"
python -m study.ask                      # interactive
python -m study.evaluate                 # reproduce the table above

set ANTHROPIC_API_KEY=your-key           # optional, Windows
python -m study.ask --claude "Why does a vaccine protect you later?"
```

## Design decisions
- **Refusal is a feature, measured separately from accuracy.** Answering an out-of-scope question counts as a failure.
- **Thresholds are tuned on one split and reported on another**, to avoid flattering numbers.
- **Extractive default:** the answer text is checked in a test to be literally quoted from the notes.
- **Citations are mandatory in Claude mode.** An uncited answer is treated as a refusal.

## Limits and next steps
- Keyword search (BM25) misses paraphrases such as "heart's pacemaker" vs "SA node". Next step: add embeddings and compare retrieval accuracy.
- Add a small web UI and deploy it.
- Grow the notes and question set, and have a second person write test questions.
- Run Claude mode against the same questions and add a row to the table.
