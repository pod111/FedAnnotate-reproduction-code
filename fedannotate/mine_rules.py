# -*- coding: utf-8 -*-
"""
mine_rules.py
=============
Mines the gazetteer and two-character suffix rules used as labeling functions.
Each corpus is mined from its own annotation text, so no rule carries
evaluation-split information.

    python -m fedannotate.mine_rules

    python -m fedannotate.mine_rules --dataset ner1998

Writes _mined_fed.json or _mined_ner1998.json next to this module. Both files
are also bundled, so this step is only needed to verify the mining.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

from . import core

TARGETS = {
    "msra": ("msra_train.txt", "msra", "_mined_fed.json", 8, 0.97, 25, 0.90),
    "ner1998": ("ner1998_train.csv", "ner1998", "_mined_ner1998.json", 2, 0.90, 4, 0.85),
}


def mine(items, min_count, min_purity, suf_min_count, suf_min_purity):
    """Return (exact gazetteer, two-character suffix rules).

    A surface form is kept only when its most frequent label clears both a
    count floor and a purity floor; the same applies to a suffix pooled over all
    surfaces that end with it.
    """
    surf = defaultdict(Counter)
    for w, t in items:
        surf[w][t] += 1

    lex = {}
    for w, c in surf.items():
        tot = sum(c.values())
        if tot < min_count:
            continue
        (bt, bc), = c.most_common(1)
        if bc / tot >= min_purity:
            lex[w] = bt

    suf = defaultdict(Counter)
    for w, c in surf.items():
        if len(w) >= 2:
            for t, n in c.items():
                suf[w[-2:]][t] += n
    suf_rules = {}
    for k, c in suf.items():
        tot = sum(c.values())
        if tot < suf_min_count:
            continue
        (bt, bc), = c.most_common(1)
        if bc / tot >= suf_min_purity:
            suf_rules[k] = bt
    return lex, suf_rules


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(TARGETS), default="msra")
    args = ap.parse_args()

    fname, reader_key, out_name, mc, mp, smc, smp = TARGETS[args.dataset]
    path = core.data_path(fname)
    if not os.path.exists(path):
        raise SystemExit(f"[skip] corpus not found: {path}")

    items = core.load_items(path, core.READERS[reader_key])
    lex, suf_rules = mine(items, mc, mp, smc, smp)

    print(f"{args.dataset}: {len(items)} annotated spans")
    print(f"  exact gazetteer : {len(lex):5d} entries  "
          f"{dict(Counter(lex.values()))}")
    print(f"  2-char suffixes : {len(suf_rules):5d} rules    "
          f"{dict(Counter(suf_rules.values()))}")

    if os.path.basename(path).startswith(("msra_test", "ner1998_dev")):
        hit = sum(1 for w, _ in items if w in lex)
        ok = sum(1 for w, t in items if lex.get(w) == t)
        print(f"  held-out coverage {hit / len(items) * 100:.1f}%  "
              f"precision when hit {ok / max(hit, 1) * 100:.1f}%")

    out = core.artifact_path(out_name)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"lexicon": lex, "suffix2": suf_rules}, fh, ensure_ascii=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
