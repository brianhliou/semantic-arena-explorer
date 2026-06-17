# Semantic Arena — One-Clue Reference

How much can a single word communicate? In word-association games like Codenames, a player must make a partner pick an exact subset of a 25-word board using one clue word plus a count. The tempting move is one clue that points at all nine of your cards at once. This repo measures whether that clue exists, and ships an explorer so you can feel where it breaks.

**Under a fixed vocabulary of 7,009 common English words and a standard embedding-similarity decoder, a single legal word pins down all nine of your cards on only 0.286% of boards — by exact count over 202 million board assignments, not a sample. The realistic ceiling is about six. The wall isn't the geometry of the embedding space, it's the vocabulary.** And the clues that *do* satisfy the decoder mostly fail on a real reader: ordinary clues recovered at 23 of 24, decoder-certified clues at 0 of 44.

## Try it

**[semantic-arena.brianhliou.com](https://semantic-arena.brianhliou.com)** — pick a target subset and watch the decoder find a clue or hit the wall, then see decoder-perfect clues that a real reader still can't read.

Run it locally with no build step:

```bash
cd explorer && python3 -m http.server 8000
# open http://localhost:8000
```

The explorer is a single static page. The browser runs the exact decoder (dot products on a shipped cosine matrix), so there is no backend.

## What's here

```
explorer/        the static one-clue explorer (index.html + precomputed data)
paper/           the full paper (PDF + LaTeX source)
reproduce/       standalone decoder + scripts to regenerate the explorer data
```

## Reproduce

The headline result is an exact enumeration; the explorer data and the
dissociation cards regenerate from the committed inputs plus the GloVe-6B slim
embedding pack (see [reproduce/DATA.md](reproduce/DATA.md) to fetch it).

```bash
cd reproduce
pip install -r requirements.txt
python3 fetch_embeddings.py          # downloads the 31MB slim pack
python3 precompute.py                # rewrites ../explorer/data/boards.json
```

The decoder is about 60 lines ([reproduce/semantic_clue/decoder.py](reproduce/semantic_clue/decoder.py)): a clue recovers a subset when, ranking board words by cosine similarity to the clue, the targets take the top spots. The projection margin is the best any legal clue does — positive means a word separates the subset, negative is the wall.

## The paper

[**Decoder-Recoverable Is Not Communicable: One-Clue Reference Under Constrained Vocabularies**](paper/one-clue-reference.pdf)

The deterministic results (coverage, frontier, the vocabulary-projection wall, the symbolic replication) are exact and reproducible here. The pragmatic-transfer results (the 0-of-44 dissociation) are pilot-scale: small audits with language-model readers standing in for humans, plus one blinded human pass. A full human panel is the test they still need, and the paper labels them as pilot throughout.

## Citation

```bibtex
@misc{liou2026oneclue,
  title  = {Decoder-Recoverable Is Not Communicable: One-Clue Reference Under Constrained Vocabularies},
  author = {Liou, Brian},
  year   = {2026},
  note   = {https://github.com/brianhliou/semantic-arena-explorer}
}
```

## License

Code and data under [MIT](LICENSE). The paper text is the author's; cite it rather than redistributing verbatim.
