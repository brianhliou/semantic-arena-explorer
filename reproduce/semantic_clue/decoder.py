"""The embedding-rank one-clue decoder, in ~60 lines.

A clue recovers a target subset when, ranking every board word by cosine
similarity to the clue, the targets occupy the top positions. The projection
margin of a subset is the best any legal clue does:

    margin(T) = max over legal clues c of
                [ min over targets t of cos(c, t) - max over non-targets r of cos(c, r) ]

Positive margin means some legal word separates the subset (recoverable);
negative means none does (the wall). Vendored from the research repo so this
package reproduces the explorer data and headline numbers with numpy alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingPack:
    """L2-normalized embedding matrix plus a parallel word list."""

    vectors: np.ndarray            # (N, dim) float32, unit-norm rows
    words: tuple[str, ...]
    word_to_index: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> "EmbeddingPack":
        data = np.load(path, allow_pickle=True)
        words = tuple(str(w) for w in data["words"])
        return cls(
            vectors=np.ascontiguousarray(data["vectors"], dtype=np.float32),
            words=words,
            word_to_index={w: i for i, w in enumerate(words)},
        )

    def index_of(self, word: str) -> int | None:
        return self.word_to_index.get(word.lower())


def projection_margin(sims_row: np.ndarray, subset_idx: list[int], n: int) -> float:
    """min target similarity minus max non-target similarity, for one clue."""
    sel = set(subset_idx)
    min_t = min(sims_row[t] for t in subset_idx)
    max_r = max(sims_row[r] for r in range(n) if r not in sel)
    return float(min_t - max_r)


def best_clue(pack: EmbeddingPack, board_words: list[str], subset: list[str],
              clue_words: list[str]) -> tuple[str, float]:
    """Return the legal clue maximizing the projection margin, and that margin.

    Because the pack is L2-normalized, cosine is a dot product, so the whole
    clue-vocabulary search is one matrix multiply.
    """
    bidx = [pack.index_of(w) for w in board_words]
    if any(i is None for i in bidx):
        raise ValueError("board word missing an embedding vector")
    board = pack.vectors[bidx]                                  # (n, dim)

    board_set = {w.lower() for w in board_words}
    legal = [(w, pack.index_of(w)) for w in clue_words if w.lower() not in board_set]
    legal = [(w, i) for w, i in legal if i is not None]
    clues = pack.vectors[[i for _, i in legal]]                 # (M, dim)

    sims = clues @ board.T                                       # (M, n) cosine
    sub_idx = [board_words.index(w) for w in subset]
    sel = np.zeros(len(board_words), dtype=bool)
    sel[sub_idx] = True
    min_t = sims[:, sel].min(axis=1)
    max_r = sims[:, ~sel].max(axis=1)
    margins = min_t - max_r
    best = int(np.argmax(margins))
    return legal[best][0], float(margins[best])
