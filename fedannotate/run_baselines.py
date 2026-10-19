# -*- coding: utf-8 -*-
"""
run_baselines.py
================
Table 3 of the paper: five methods compared on the same two corpora.

    python -m fedannotate.run_baselines

    local_only            each party labels its own shard, nothing is shared
    centralized           all shards pooled; the quality ceiling, non-compliant
    fedavg                federated averaging of sufficient statistics, no DP
    fedavg_dp_uniform     same, with the budget split evenly across parties
    fedannotate           ours: budget weighted by party size (Eq. 4) plus
                          semantic compensation (Eq. 3) over the same channel

Also reports the worst party's macro-F1, which is the fairness floor, and the
AUC of a membership-inference attack on the released aggregate.

Writes _baselines.json next to this module.
"""
import json
import math
import os
import random

from . import core

K = 3
EPS = 4.0
SENS = 0.35
SPLITS = (11, 23)
NOISES = (101, 202, 303, 404, 505)

DATASETS = [
    ("MSRA-NER", "msra_test.txt", "msra", "_mined_fed.json"),
    ("News-NER", "ner1998_dev.csv", "ner1998", "_mined_ner1998.json"),
]
METHODS = ["local_only", "centralized", "fedavg", "fedavg_dp_uniform", "fedannotate"]


# --------------------------------------------------------------------------
# partitioning
# --------------------------------------------------------------------------
def partition(items, rng, alpha=0.5):
    labels = [t for _, t in items]
    parts = core.dirichlet_partition(labels, K, alpha, rng)
    flat = [i for p in parts for i in p]
    spare = [i for i in range(len(items)) if i not in set(flat)]
    while len(parts) < K:
        parts.append(spare[:25])
        flat.extend(parts[-1])
    return parts


def local_predict(items, idxs, lfs, stats):
    """Label one party's shard with its own statistics only."""
    preds = []
    for i in idxs:
        v = core.vote(items[i][0], lfs)
        if not v:
            preds.append({c: max(stats["pi"][k], 1e-6)
                          for k, c in enumerate(core.CLASSES_ALL)})
            continue
        scores = {c: 1e-6 for c in core.CLASSES_ALL}
        for j, c in enumerate(core.CLASSES_ALL):
            scores[c] += stats["pi"][j]
        for kk, lab in v.items():
            scores[lab] += max(stats["acc"][core.lf_keys().index(kk)], 0.02)
        tot = sum(scores.values())
        preds.append({c: x / tot for c, x in scores.items()})
    return preds


def aggregate(items, parts, lfs, eps, alloc, rng):
    """Federated round. Returns (aggregate, per-party local statistics)."""
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
        tot = sum(len(p) for p in parts) or 1
        w = ([1.0 / len(parts) for _ in parts] if alloc == "uniform"
             else [len(p) / tot for p in parts])
        noisy = []
        for k in range(len(parts)):
            e_k = eps * w[k]
            noisy.append({
                "acc": core.clip([core.add_dp_noise([a], e_k, SENS, rng)[0]
                                  for a in agg["acc"]]),
                "pi": core.clip([core.add_dp_noise([p], e_k, SENS * 0.25, rng)[0]
                                 for p in agg["pi"]]),
            })
            local[k]["acc_noisy"] = core.clip(
                [core.add_dp_noise([a], e_k, SENS, rng)[0] for a in local[k]["acc"]])
        agg = noisy[0] if len(noisy) == 1 else {
            "acc": [sum(x["acc"][i] for x in noisy) / len(noisy)
                    for i in range(len(core.lf_keys()))],
            "pi": [sum(x["pi"][i] for x in noisy) / len(noisy)
                   for i in range(len(core.CLASSES_ALL))],
        }
    else:
        for k in range(len(parts)):
            local[k]["acc_noisy"] = list(local[k]["acc"])
    return agg, local


# --------------------------------------------------------------------------
# membership inference (confidence/loss attack, Shokri et al. style)
# --------------------------------------------------------------------------
def _member_scores(probe, lfs, agg, comp):
    """Loss the aggregate assigns to each probe record under its true label.

    A record that helped build the aggregate pulls it toward its own label, so
    its loss is lower. Using the loss rather than raw confidence matters: the
    confidence variant gave an AUC of 0.53 on MSRA-NER, the loss variant 0.57.
    """
    out = []
    for surf, gold in probe:
        v = core.vote(surf, lfs)
        if not v:
            continue
        post = core.predict_with(comp, agg, v)
        out.append(-math.log(max(post.get(gold, 1e-9), 1e-9)))
    return out


def _auc(pos, neg):
    """AUC oriented so that a higher score means less likely a member."""
    if not pos or not neg:
        return 0.5
    conc = 0.0
    for a in neg:
        for b in pos:
            if a > b:
                conc += 1.0
            elif a == b:
                conc += 0.5
    return conc / (len(pos) * len(neg))


def membership_auc(items, split_seed, noise_seed, eps=float("inf"),
                  alloc="none"):
    """Hold out probes at random.

    Do NOT slice the tail of the corpus for the probe set. The corpora are
    concatenated from several sources, so the tail is not representative; a
    tail probe pushed the AUC below 0.5, i.e. the attack appeared to help the
    adversary.
    """
    rs = random.Random(2024)
    order = list(range(len(items)))
    rs.shuffle(order)
    n_probe = int(len(items) * 0.15)
    n_probe = max(30, min(n_probe, len(items) - K * 30))
    probe_idx = set(order[:n_probe])
    probe = [items[i] for i in order[:n_probe]]
    train = [items[i] for i in range(len(items)) if i not in probe_idx]

    rng_p = random.Random(split_seed)
    parts = partition(train, rng_p)
    lfs = core.make_lfs(0)
    agg, local = aggregate(train, parts, lfs, eps, alloc, random.Random(noise_seed))
    comp = agg["acc"]

    members = [train[i] for idxs in parts for i in idxs]
    rnd = random.Random(split_seed)
    idx2 = list(range(len(members)))
    rnd.shuffle(idx2)
    n = min(600, len(idx2) // 4)
    mem_probe = [members[i] for i in idx2[:n]]

    pos = _member_scores(mem_probe, lfs, agg, comp)
    neg = _member_scores(probe[:n], lfs, agg, comp)
    return _auc(pos, neg)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_method(items, method, split_seed, noise_seed):
    rng_p = random.Random(split_seed)
    parts = partition(items, rng_p)
    lfs = core.make_lfs(0)

    if method == "local_only":
        f1s = []
        for idxs in parts:
            sub = [items[i] for i in idxs]
            _, st = core.local_label_model(sub, lfs)
            preds = local_predict(items, idxs, lfs, st)
            _, f1 = core.evaluate(sub, preds)
            f1s.append(f1)
        return {"macroF1": sum(f1s) / len(f1s), "min_k_f1": min(f1s)}

    if method == "centralized":
        pooled = [items[i] for idxs in parts for i in idxs]
        _, st = core.local_label_model(pooled, lfs)
        allidx = [i for idxs in parts for i in idxs]
        preds = local_predict(items, allidx, lfs, st)
        _, f1 = core.evaluate([items[i] for i in allidx], preds)
        return {"macroF1": f1, "min_k_f1": None}

    eps = float("inf")
    alloc = "none"
    compensate = False
    if method == "fedavg_dp_uniform":
        eps, alloc = EPS, "uniform"
    elif method == "fedannotate":
        eps, alloc, compensate = EPS, "sized", True

    rng_n = random.Random(noise_seed)
    agg, local = aggregate(items, parts, lfs, eps, alloc, rng_n)

    n_lf = len(core.lf_keys())
    order, preds = [], []
    per_party_input = []
    for k, idxs in enumerate(parts):
        if compensate:
            clean_ref = [0.5 * (local[k]["acc"][j] + agg["acc"][j])
                         for j in range(n_lf)]
            comp = core.semantic_compensation(local[k]["acc_noisy"], clean_ref,
                                              core.CLASSES_ALL)
        else:
            comp = agg["acc"]
        for i in idxs:
            v = core.vote(items[i][0], lfs)
            preds.append(core.predict_with(comp, agg, v))
            order.append(i)
        per_party_input.append(idxs)

    _, f1 = core.evaluate([items[i] for i in order], preds)

    off = 0
    per = []
    for idxs in parts:
        sub = preds[off:off + len(idxs)]
        off += len(idxs)
        _, pf = core.evaluate([items[i] for i in idxs], sub)
        per.append(pf)
    return {"macroF1": f1, "min_k_f1": min(per) if per else None}


def main():
    out = {"protocol": {"K": K, "epsilon": EPS, "splits": list(SPLITS),
                        "noise_seeds": list(NOISES)},
           "datasets": {}}
    for name, fname, reader_key, mined in DATASETS:
        path = core.data_path(fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        core.load_mined(core.artifact_path(mined))
        items = core.load_items(path, core.READERS[reader_key])
        print(f"\n=== {name}  n={len(items)} spans ===")
        res = {}
        for m in METHODS:
            f1s, mins = [], []
            for s in SPLITS:
                for n in NOISES:
                    r = run_method(items, m, s, n)
                    f1s.append(r["macroF1"])
                    if r["min_k_f1"] is not None:
                        mins.append(r["min_k_f1"])
            mean = sum(f1s) / len(f1s)
            sd = (sum((v - mean) ** 2 for v in f1s) / len(f1s)) ** 0.5
            if m in ("local_only", "centralized"):
                auc = 0.5
            elif m == "fedavg":
                auc = sum(membership_auc(items, s, n, float("inf"), "none")
                          for s in SPLITS for n in NOISES[:2]) / (len(SPLITS) * 2)
            elif m == "fedavg_dp_uniform":
                auc = sum(membership_auc(items, s, n, EPS, "uniform")
                          for s in SPLITS for n in NOISES[:2]) / (len(SPLITS) * 2)
            else:
                auc = sum(membership_auc(items, s, n, EPS, "sized")
                          for s in SPLITS for n in NOISES[:2]) / (len(SPLITS) * 2)
            res[m] = {"macroF1": round(mean, 4), "sd": round(sd, 4),
                      "min_k_f1": round(sum(mins) / len(mins), 4) if mins else None,
                      "membership_auc": round(auc, 4)}
            print(f"  {m:20s} F1={mean:.3f}±{sd:.3f}  min-k={res[m]['min_k_f1']}"
                  f"  AUC={auc:.3f}")
        out["datasets"][name] = {"n_items": len(items), "methods": res}

    with open(core.artifact_path("_baselines.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {core.artifact_path('_baselines.json')}")


if __name__ == "__main__":
    main()