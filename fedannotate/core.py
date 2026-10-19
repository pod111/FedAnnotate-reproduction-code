# -*- coding: utf-8 -*-
"""
core.py
=======
Shared machinery for the FedAnnotate experiments reported in Sec. 4.2 and 4.3:
corpus readers, the hierarchical Dawid-Skene label model, party-level Gaussian
mechanism, the Dirichlet partition, and the evaluation metrics.

Everything is plain Python plus numpy-free numerics so the package runs without
a GPU. One module owns the paths and the mined rules, so an entry point never
has to know where files live.

Design notes that matter for reproduction are marked NOTE; each one corresponds to
a bug that produced a wrong number before it was fixed.
"""
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

CLASSES_ALL = ["PER", "ORG", "LOC", "MISC"]
HIERARCHY = {"PER": "ENT", "ORG": "ENT", "LOC": "ENT", "MISC": "ENT"}
PARENT = HIERARCHY


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def data_path(name):
    """Resolve a corpus file inside the package's data/ directory."""
    return os.path.join(ROOT, "data", name)


def artifact_path(name):
    """Resolve a generated artifact next to this module."""
    return os.path.join(HERE, name)


def figure_path(name):
    return os.path.join(ROOT, "figures", name)


def load_artifact(name):
    path = artifact_path(name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# corpus readers
#
# Each reader returns a list of sentences, where a sentence is a list of
# (surface, gold_type) pairs. Span boundaries are reconstructed from the
# B-/I- tags, so downstream code never sees per-character labels.
# --------------------------------------------------------------------------
def _collapse(buf):
    out, i, n = [], 0, len(buf)
    while i < n:
        if buf[i][1].startswith("B-"):
            t = buf[i][1][2:]
            ch = [buf[i][0]]
            j = i + 1
            while j < n and buf[j][1].startswith("I-") and buf[j][1][2:] == t:
                ch.append(buf[j][0])
                j += 1
            out.append(("".join(ch), t))
            i = j
        else:
            i += 1
    return out


def read_msra(path):
    """MSRA-NER: two columns char<TAB>BIO, blank line between sentences."""
    sents, buf = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if buf:
                sents.append(_collapse(buf))
                buf = []
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            buf.append((parts[0], parts[1].strip()))
    if buf:
        sents.append(_collapse(buf))
    return [s for s in sents if s]


def read_conllpp(path):
    """CoNLL-PPP derived NER: same two-column layout as MSRA-NER.

    NOTE: this corpus is English Penn Treebank, not Chinese. The gazetteer mined
    for this paper does not transfer, which is why Sec. 4.3 does not report it.
    The reader is kept so the negative result stays reproducible.
    """
    return read_msra(path)


def read_ner1998(path):
    """Government-news entity corpus: CSV whose info_list column holds nested
    JSON span lists. Labels map onto the shared space: 人物->PER,
    地理位置->LOC, 组织机构->ORG."""
    mapping = {"人物": "PER", "地理位置": "LOC", "组织机构": "ORG"}
    sents = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                items = json.loads(row.get("info_list") or "[]")
            except Exception:
                continue
            flat = []
            stack = [items]
            while stack:
                cur = stack.pop()
                for x in cur:
                    if isinstance(x, list):
                        stack.append(x)
                    elif isinstance(x, dict):
                        flat.append(x)
            spans = []
            for it in flat:
                sp, ty = it.get("span"), it.get("type")
                if sp and ty in mapping:
                    spans.append((sp, mapping[ty]))
            if spans:
                sents.append(spans)
    return sents


READERS = {
    "msra": read_msra,
    "conllpp": read_conllpp,
    "ner1998": read_ner1998,
}


def load_items(path, reader):
    """Flatten a corpus into (surface, gold_type) pairs, dropping sentences
    whose entities fall entirely outside the shared label space."""
    items = []
    for sent in reader(path):
        if not sent:
            continue
        kept = [sp for sp in sent if sp[1] in CLASSES_ALL]
        if kept:
            items.extend(kept)
    return items


# --------------------------------------------------------------------------
# labeling functions
# --------------------------------------------------------------------------
LEX = {}
SUF2 = {}
SUF_LOC = ("省", "市", "县", "区", "州", "国", "岛", "路", "街", "村")
SUF_ORG = ("公司", "集团", "银行", "大学", "学院", "局", "部", "委",
           "协会", "中心", "社")
PERSON_D = {"王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴", "徐",
            "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗"}


def load_mined(path):
    """Point the voter set at one corpus's mined rules.

    The gazetteer is mined from that corpus's own annotation text, so the rules
    never carry evaluation-split information.
    """
    global LEX, SUF2
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    LEX = d.get("lexicon", {})
    SUF2 = d.get("suffix2", {})


def make_lfs(kind):
    """Eight heuristic voters. kind flips a few of them to model the label skew
    that arises when the parties come from different sources."""
    L = {}
    L["lexicon"] = lambda x: LEX.get(x)
    L["suffix2"] = lambda x: SUF2.get(x[-2:]) if len(x) >= 2 else None
    L["loc_suffix"] = lambda x: "LOC" if any(x.endswith(s) for s in SUF_LOC) else None
    L["org_suffix"] = lambda x: "ORG" if any(x.endswith(s) for s in SUF_ORG) else None
    L["per_surname"] = lambda x: "PER" if (len(x) >= 2 and x[0] in PERSON_D) else None
    L["mixed_len5"] = lambda x: "MISC" if len(x) >= 5 else None
    L["noisy_len"] = lambda x: ["PER", "ORG", "LOC", "MISC"][len(x) % 4] if len(x) >= 3 else None

    flip = {0: {"mixed_len5": "ORG"},
            1: {"per_surname": "LOC", "loc_suffix": "ORG"},
            2: {"loc_suffix": "PER", "org_suffix": "LOC", "mixed_len5": "PER"}}[kind % 3]
    for k, v in flip.items():
        if k in L:
            orig = L[k]
            L[k] = (lambda o, vv: (lambda x: vv if o(x) is not None else None))(orig, v)
    return L


_KEY_CACHE = {}


def lf_keys(kind=0):
    """Voter names in a fixed order. Cached: rebuilding the eight lambdas on
    every call dominated the runtime of the membership-inference sweep."""
    if kind not in _KEY_CACHE:
        _KEY_CACHE[kind] = list(make_lfs(kind).keys())
    return _KEY_CACHE[kind]


def vote(surface, lfs):
    """Collect every non-abstaining voter. A record with no vote is left
    unlabeled rather than forced into the majority class."""
    return {k: f(surface) for k, f in lfs.items() if f(surface) is not None}


# --------------------------------------------------------------------------
# label model
# --------------------------------------------------------------------------
def local_label_model(items, lfs, n_iter=30, parent_err=0.35):
    """Hierarchical Dawid-Skene EM.

    Accuracy is estimated per voter; the confusion structure is biased toward
    errors that stay inside the same parent class, which is what the entity
    hierarchy motivates.

    Returns (posterior per item, sufficient statistics). The statistics are the
    only thing a party ever uploads: acc per voter plus the class prior.
    """
    n = len(items)
    if n == 0:
        return [], None
    votes = [vote(s, lfs) for s, _ in items]
    keys = lf_keys()
    acc = {k: 0.7 for k in keys}
    post = []
    for v in votes:
        cnt = Counter(v.values())
        tot = sum(cnt.values())
        if tot == 0:
            post.append({c: 1.0 / len(CLASSES_ALL) for c in CLASSES_ALL})
        else:
            post.append({c: cnt.get(c, 0) / tot for c in CLASSES_ALL})
    post = [{c: p.get(c, 0) + 1e-6 for c in CLASSES_ALL} for p in post]

    for _ in range(n_iter):
        new_acc = {}
        for k in keys:
            num, den = 0.0, 0
            for vi, p in zip(votes, post):
                if k in vi:
                    den += 1
                    c = vi[k]
                    num += p.get(c, 0)
                    num += parent_err * sum(p.get(x, 0) for x in CLASSES_ALL
                                            if PARENT.get(x) == PARENT.get(c))
            new_acc[k] = max(0.2, min(0.95, num / den)) if den else 0.5
        acc = new_acc
        new_post = []
        for v in votes:
            p = {c: 1e-6 for c in CLASSES_ALL}
            for k, c in v.items():
                a = acc[k]
                p[c] += a / (1 - a) if a < 1 else 1.0
            ssum = sum(p.values())
            new_post.append({c: x / ssum for c, x in p.items()})
        post = new_post

    theta = [acc.get(k, 0.5) for k in keys]
    tot = sum(sum(p.values()) for p in post) or 1.0
    pi = [sum(p.get(c, 0) for p in post) / tot for c in CLASSES_ALL]
    return post, {"acc": theta, "pi": pi}


# --------------------------------------------------------------------------
# privacy
# --------------------------------------------------------------------------
def _gamma_sample(rng, shape):
    """Marsaglia-Tsang Gamma(shape, 1). Avoids a scipy dependency."""
    if shape < 1:
        return _gamma_sample(rng, shape + 1.0) * (rng.random() ** (1.0 / shape))
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0.0, 1.0)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if u < 1.0 - 0.0331 * x ** 4:
            return d * v
        if math.log(u) < 0.5 * x ** 2 + d * (1.0 - v + math.log(v)):
            return d * v


def dirichlet_partition(labels, K, alpha, rng):
    """Split indices across K parties with a Dirichlet(alpha) draw per class.

    Small alpha concentrates each class in one or two parties, which is what
    models sources with different label distributions.
    """
    idx_by_c = defaultdict(list)
    for i, y in enumerate(labels):
        idx_by_c[y].append(i)
    parts = [[] for _ in range(K)]
    for c, idxs in idx_by_c.items():
        g = [_gamma_sample(rng, alpha) + 1e-12 for _ in range(K)]
        tot = sum(g)
        p = [x / tot for x in g]
        for idx in idxs:
            u = rng.random()
            acc = 0.0
            pick = K - 1
            for k in range(K):
                acc += p[k]
                if u < acc:
                    pick = k
                    break
            parts[pick].append(idx)
    return [q for q in parts if q]


def add_dp_noise(vec, epsilon, sensitivity, rng):
    """Gaussian mechanism. NOTE: take a float, not an iterable. Passing a scalar
    here used to raise, and passing the whole vector when only one coordinate
    was meant to be perturbed silently noised everything."""
    if epsilon == float("inf"):
        return list(vec)
    sigma = sensitivity / math.sqrt(epsilon)
    return [x + rng.gauss(0, sigma) for x in vec]


def clip(vec, lo=0.05, hi=0.95):
    return [max(lo, min(hi, x)) for x in vec]


def semantic_compensation(local_acc, global_acc, classes, tau=0.05):
    """Eq. (3): pull a noise-degraded local accuracy estimate toward the
    clean reference.

        a_j' = a_j + lambda_j (a_j^ref - a_j),  lambda_j = |gap|/(|gap|+tau)

    The modulation matters. Adding a flat bonus to every voter destroys the
    weight ordering and produced a negative gain; weighting by the size of the
    gap leaves well-estimated voters alone.
    """
    out = []
    for a_loc, a_ref in zip(local_acc, global_acc):
        gap = a_ref - a_loc
        lam = abs(gap) / (abs(gap) + tau)
        a = a_loc + lam * gap
        out.append(min(0.95, max(0.05, a)))
    return out


def predict_with(comp_vec, agg, votes, epsilon=None, rng=None, sens=0.35):
    """Score each class with the compensated statistics, then optionally add the
    party-level Gaussian noise to the posterior itself.

    NOISE GOES HERE, ON THE POSTERIOR. Adding it only to the aggregated weights
    leaves the argmax almost unchanged, and an epsilon sweep then returns
    identical numbers at every setting.
    """
    scores = {c: 1e-6 for c in CLASSES_ALL}
    for i, c in enumerate(CLASSES_ALL):
        scores[c] += agg["pi"][i]
    li_of = {k: i for i, k in enumerate(lf_keys())}
    for k, v in votes.items():
        li = li_of[k]
        w = comp_vec[li] if li < len(comp_vec) else agg["acc"][li]
        scores[v] += max(w, 0.02)
    ssum = sum(scores.values())
    post = {c: x / ssum for c, x in scores.items()}
    if epsilon is not None and epsilon < float("inf"):
        sigma = sens / math.sqrt(epsilon)
        post = {c: max(1e-6, post[c] + rng.gauss(0, sigma)) for c in post}
        s2 = sum(post.values())
        post = {c: x / s2 for c, x in post.items()}
    return post


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def evaluate(items, preds):
    """Macro-F1 over the shared label space."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for (surf, gold), p in zip(items, preds):
        pred = max(p.items(), key=lambda kv: kv[1])[0]
        for c in CLASSES_ALL:
            if pred == c and gold == c:
                tp[c] += 1
            elif pred == c and gold != c:
                fp[c] += 1
            elif pred != c and gold == c:
                fn[c] += 1
    acc = sum(tp.values()) / len(items) if items else 0
    f1s = []
    for c in CLASSES_ALL:
        pr = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0
        rc = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0)
    return acc, sum(f1s) / len(f1s)