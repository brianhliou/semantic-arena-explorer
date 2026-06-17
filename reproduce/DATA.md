# Data

Small inputs are committed; the embedding pack is fetched.

## Committed

```
data/words/wordnet-concrete-v0.1.json       25-word board source (concrete nouns)
data/clue-vocabs/wordnet-brown-common-v0.1.json   7,009 legal clue words
data/audit/*.jsonl                          blinded reader-audit logs (controls + challenge)
```

The audit logs carry, per item, the board, the clue and count, the
decoder-certified subset (`model_subset`) and its margin (`model_margin`), and
the reader's actual ordered guesses (`response.guesses`). The dissociation view
is built straight from these.

## Embedding pack (fetched, not committed)

`glove-6B-300d-slim.npz` (~31 MB) is the 300-dimensional GloVe-6B vectors
restricted to the working vocabulary and L2-normalized. It is too large to
commit, so fetch it from the latest release:

```bash
python3 fetch_embeddings.py
```

This writes `data/embeddings/glove-6B-300d-slim.npz`. Once present,
`python3 precompute.py` regenerates `../explorer/data/boards.json`.

### Rebuilding from source instead

The pack is GloVe-6B (Stanford NLP, [nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)),
filtered to the union of the board-word and clue vocabularies above and
L2-normalized per row. Any rebuild that keeps those two properties (same
vocabulary, unit-norm rows) reproduces the explorer and the headline numbers.
