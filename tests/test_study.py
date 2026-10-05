from types import SimpleNamespace

from study.answer import answer, claude_answer, gate, load_index
from study.index import load_chunks, tokenize
from study.answer import NOTES

INDEX = load_index()


def fake_client(text):
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)])))


def test_notes_are_split_into_cited_sections():
    chunks = load_chunks(NOTES)
    assert len(chunks) == 30
    assert all(c.doc and c.section and c.text for c in chunks)


def test_answers_cite_the_right_note():
    a = answer("Which valve sits between the left atrium and left ventricle?", INDEX)
    assert a.answered and any(c.startswith("heart >") for c in a.citations)
    assert "mitral" in a.text.lower()


def test_refuses_out_of_domain_question():
    a = answer("What is the capital of France?", INDEX)
    assert not a.answered and a.citations == []


def test_refuses_topic_not_in_notes():
    assert not answer("What are the layers of the skin?", INDEX).answered


def test_answer_text_is_quoted_from_notes_only():
    a = answer("What is the functional unit of the kidney?", INDEX)
    notes = " ".join(c.text for c in load_chunks(NOTES))
    for sentence in a.text.strip('"').split('" "'):
        assert sentence.strip('"') in notes


def test_stemming_and_stopwords():
    assert tokenize("What are the lobes?") == ["lobe"]


def test_claude_mode_maps_citations_to_sources():
    hits, _ = gate(INDEX, "Which gland is known as the master gland?")
    a = claude_answer("q", hits, fake_client("The pituitary gland [1]."))
    assert a.answered and a.citations == [hits[0][1].cite]


def test_claude_mode_honours_not_in_notes():
    hits, _ = gate(INDEX, "Which gland is known as the master gland?")
    assert not claude_answer("q", hits, fake_client("NOT_IN_NOTES")).answered


def test_claude_answer_without_citation_is_rejected():
    hits, _ = gate(INDEX, "Which gland is known as the master gland?")
    assert not claude_answer("q", hits, fake_client("The pituitary gland.")).answered
