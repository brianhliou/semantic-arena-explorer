#!/usr/bin/env python3
"""Regenerate the explorer's data file (../explorer/data/boards.json).

For each board we ship the 25 board words plus the clue-to-board cosine matrix
for every legal clue, so the browser runs the exact embedding-rank decoder
client-side. We also extract the decoder-certified-but-reader-fails examples
(and passing controls) for the dissociation view, straight from the audit logs.

Needs the GloVe-6B slim embedding pack; see DATA.md to fetch it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from semantic_clue.decoder import EmbeddingPack  # noqa: E402

REPO = HERE.parent
EMBEDDINGS = HERE / "data" / "embeddings" / "glove-6B-300d-slim.npz"
CLUE_VOCAB = HERE / "data" / "clue-vocabs" / "wordnet-brown-common-v0.1.json"
# Certified-challenge items come from the audit log; controls come from the
# randomized-position (shuffled) control run.
CERT_FILES = [HERE / "data" / "audit" / "openai-nano-challenge-v0.1.jsonl"]
CONTROL_FILES = [HERE / "data" / "audit" / "shuffled-controls-gpt-5.4-nano-v0.2.jsonl"]
OUT = REPO / "explorer" / "data" / "boards.json"

# The darling counterexample board: the full 9-key is unrecoverable
# (margin -0.069), but small clean subsets are, so one board shows both states.
DARLING_BOARD = [
    "specialist", "machinery", "mud", "wash", "maid", "horizon", "sandwich",
    "wood", "call", "barn", "product", "pit", "stone", "submarine", "prison",
    "tent", "educator", "combustion", "colonel", "ivory", "cotton",
    "processing", "builder", "listener", "scholar",
]
DARLING_SUBSET = [
    "barn", "builder", "call", "colonel", "educator", "pit", "product",
    "specialist", "wood",
]
# Decoder-certified clues (positive margin) on which a real reader still fails.
CERT_CLUES = ["vain", "carriage", "reminder", "colony"]
SIM_FLOOR = 0.02   # a clue below this to every board word can never win a subset


def build_board(board_id, words, headline_subset, pack, vocab):
    bidx = [pack.index_of(w) for w in words]
    missing = [w for w, i in zip(words, bidx) if i is None]
    if missing:
        raise SystemExit(f"board {board_id} missing vectors: {missing}")
    board_vecs = pack.vectors[bidx]
    board_set = {w.lower() for w in words}
    legal = [(w, pack.index_of(w)) for w in vocab if w.lower() not in board_set]
    legal = [(w, i) for w, i in legal if i is not None]
    sims = pack.vectors[[i for _, i in legal]] @ board_vecs.T
    keep = [o for o in np.argsort(-sims.max(axis=1)) if sims.max(axis=1)[o] >= SIM_FLOOR]
    return {
        "id": board_id,
        "words": list(words),
        "headline_subset": headline_subset,
        "clues": [legal[o][0] for o in keep],
        "sims": np.round(sims[keep], 4).tolist(),
    }


def extract_dissociation():
    cert, controls = {}, []
    for path in CERT_FILES:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            it, sc, resp = r["item"], r["scores"], r["response"]
            margin = it.get("model_margin")
            if (margin is not None and margin > 0
                    and it["clue"] in CERT_CLUES and it["clue"] not in cert):
                cert[it["clue"]] = {
                    "kind": "certified", "clue": it["clue"], "count": it["count"],
                    "margin": round(margin, 4), "model": r["model"],
                    "board_words": it["board_words"],
                    "intended": it.get("model_subset") or it["target_set"],
                    "guesses": resp["guesses"],
                    "hits": sc["target_hits"], "recovered": sc["exact_match"],
                }
    near_miss = []
    for path in CONTROL_FILES:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            it, sc, resp = r["item"], r["scores"], r["response"]
            if not it.get("control_type"):
                continue
            card = {
                "kind": "control", "clue": it["clue"], "count": it["count"],
                "margin": None, "model": r["model"],
                "board_words": it["board_words"], "intended": it["target_set"],
                "guesses": resp["guesses"],
                "hits": sc["target_hits"], "recovered": sc["exact_match"],
            }
            if sc["exact_match"] and len(controls) < 2:
                controls.append(card)
            elif not sc["exact_match"] and it["clue"] == "car" and not near_miss:
                near_miss.append(card)
    return controls + near_miss + [cert[c] for c in CERT_CLUES if c in cert]


def main() -> int:
    if not EMBEDDINGS.exists():
        raise SystemExit(f"embedding pack not found at {EMBEDDINGS}\nSee DATA.md to fetch it.")
    pack = EmbeddingPack.load(EMBEDDINGS)
    vocab = json.loads(CLUE_VOCAB.read_text())["words"]
    payload = {
        "decoder": "embedding-rank top-k, glove-6B-300d-slim",
        "sim_floor": SIM_FLOOR,
        "boards": [build_board("darling", DARLING_BOARD, DARLING_SUBSET, pack, vocab)],
        "dissociation": extract_dissociation(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT}  ({len(payload['boards'])} board, "
          f"{len(payload['boards'][0]['clues'])} clues, "
          f"{len(payload['dissociation'])} dissociation cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
