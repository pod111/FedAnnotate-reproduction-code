# -*- coding: utf-8 -*-
"""
run_tradeoff.py
===============
Table 2 and Figure 4(a)(b) of the paper: the privacy-utility trade-off on two
public Chinese corpora.

    python -m fedannotate.run_tradeoff

Protocol (identical to Sec. 4.3):
    K = 3 parties
    alpha in {1.0, 0.3}  Dirichlet concentration, controls label skew
    epsilon from 64 down to 2, party-level Gaussian mechanism
    2 split seeds x 5 noise seeds, mean reported

Writes _tradeoff.json next to this module. The bundled _news_scan.json holds
the same numbers from the run that produced the paper, so the two can be
compared directly.
"""
import json
import os
import random

from . import core

ALPHAS = (1.0, 0.3)
EPSILONS = (64, 32, 16, 8, 4, 2)
SPLITS = (11, 23)
NOISES = (101, 202, 303, 404, 505)
K = 3
SENS = 0.35

DATASETS = [
    ("MSRA-NER", "msra_test.txt", "msra", "_mined_fed.json"),
    ("News-NER", "ner1998_dev.csv", "ner1998", "_mined_ner1998.json"),
]


def one_run(items, labels, alpha, eps, split_seed, noise_seed):
    rng_p = random.Random(split_seed)
    rng_n = random.Random(noise_seed)
    parts = core.dirichlet_partition(labels, K, alpha, rng_p)
    # Keep every party trainable: on a small corpus Dirichlet can starve one.
    flat = [i for p in parts for i in p]
    spare = [i for i in range(len(items)) if i not in set(flat)]
    while len(parts) < K:
        parts.append(spare[:25])
        flat.extend(parts[-1])

    lfs = core.make_lfs(0)
    local = []
    for idxs in parts:
        sub = [items[i] for i in idxs]
        _, st = core.local_label_model(sub, lfs)
        local.append({"acc": st["acc"], "pi": st["pi"]})

    agg = {"acc": [0.0] * len(core.lf_keys()),
           "pi": [0.0] * len(core.CLASSES_ALL)}
    for st in local:
        for i, a in enumerate(st["acc"]):
            agg["acc"][i] += a / len(parts)
        for i, pv in enumerate(st["pi"]):
            agg["pi"][i] += pv / len(parts)

    if eps < float("inf"):
        # NOTE: the noise is applied to each party's own contribution, so both
        # the aggregate and the local estimate are perturbed. That is what Eq. (3)
        # assumes. If only the aggregate is noised, the local estimate stays
        # clean and the compensation has nothing to correct.
        tot = sum(len(p) for p in parts) or 1
        w = [len(p) / tot for p in parts]
        noisy = []
        for k in range(len(parts)):
            e_k = eps * w[k]
            noisy.append({
                "acc": core.clip([core.add_dp_noise([a], e_k, SENS, rng_n)[0]
                                  for a in agg["acc"]]),
                "pi": core.clip([core.add_dp_noise([p], e_k, SENS * 0.25, rng_n)[0]
                                 for p in agg["pi"]]),
            })
            local[k]["acc_noisy"] = core.clip(
                [core.add_dp_noise([a], e_k, SENS, rng_n)[0] for a in local[k]["acc"]])
        agg = noisy[0] if len(noisy) == 1 else {
            "acc": [sum(x["acc"][i] for x in noisy) / len(noisy)
                    for i in range(len(core.lf_keys()))],
            "pi": [sum(x["pi"][i] for x in noisy) / len(noisy)
                   for i in range(len(core.CLASSES_ALL))],
        }
    else:
        for k in range(len(parts)):
            local[k]["acc_noisy"] = list(local[k]["acc"])

    order, preds = [], []
    n_lf = len(core.lf_keys())
    for k, idxs in enumerate(parts):
        clean_ref = [0.5 * (local[k]["acc"][j] + agg["acc"][j]) for j in range(n_lf)]
        comp = core.semantic_compensation(local[k]["acc_noisy"], clean_ref,
                                          core.CLASSES_ALL)
        for i in idxs:
            v = core.vote(items[i][0], lfs)
            preds.append(core.predict_with(comp, agg, v, epsilon=eps,
                                           rng=rng_n, sens=SENS))
            order.append(i)
    _, f1 = core.evaluate([items[i] for i in order], preds)
    return f1


def main():
    out = {"protocol": {"K": K, "alphas": ALPHAS, "epsilons": list(EPSILONS),
                        "splits": list(SPLITS), "noise_seeds": list(NOISES)},
           "rows": []}
    for name, fname, reader_key, mined in DATASETS:
        path = core.data_path(fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        core.load_mined(core.artifact_path(mined))
        items = core.load_items(path, core.READERS[reader_key])
        labels = [t for _, t in items]
        print(f"\n=== {name}  n={len(items)} spans ===")

        for alpha in ALPHAS:
            scan = []
            for eps in EPSILONS:
                vals = [one_run(items, labels, alpha, eps, s, n)
                        for s in SPLITS for n in NOISES]
                mean = sum(vals) / len(vals)
                sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                scan.append({"epsilon": eps, "macroF1": round(mean, 4),
                             "sd": round(sd, 4)})
                print(f"  alpha={alpha} eps={eps:2d}: {mean:.3f} ± {sd:.3f}")
            out["rows"].append({"dataset": name, "alpha": alpha, "scan": scan})

    with open(core.artifact_path("_tradeoff.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {core.artifact_path('_tradeoff.json')}")


if __name__ == "__main__":
    main()