# Data directory

The corpora are **not redistributed with this package**. Place the four files
below here before running the experiments.

```
data/
  msra_train.txt        MSRA-NER training split      (mining source)
  msra_test.txt         MSRA-NER test split,6 181 spans   (evaluation)
  ner1998_train.csv     government-news training split    (mining source)
  ner1998_dev.csv       government-news eval split, 194 spans (evaluation)
```

## MSRA-NER

Microsoft Research Chinese news NER, three classes (PER / LOC / ORG).
Available from ModelScope as `iic/msra_ner`, which hosts the original files.

Format: two tab-separated columns, `char<TAB>LABEL`, blank line between
sentences. Labels are `O`, `B-PER`/`I-PER`, `B-LOC`/`I-LOC`, `B-ORG`/`I-ORG`.

## Government-news entity corpus

A Chinese news entity corpus labelled 人物 / 地理位置 / 组织机构, mapped onto
PER / LOC / ORG. CSV with columns `id,text,info_list,schema`, where
`info_list` holds a **doubly nested** JSON list of
`{"type": ..., "span": ..., "offset": [...]}` objects. `core.read_ner1998`
flattens it.

Check that the two splits do not overlap before use:

```python
import csv
ids = {s: {r["id"] for r in csv.DictReader(open(f"data/ner1998_{s}.csv",
                                                encoding="utf-8"))}
       for s in ("train", "dev", "test")}
print(len(ids["dev"] & ids["test"]))   # must be 0
```

If that intersection is non-zero, the dev and test files are the same batch
under different names; keep only one.

## A note on CoNLL-PPP

`core.read_conllpp` is provided so that the negative result reported in the
paper stays reproducible, but **no CoNLL-PPP number appears in Sec. 4.3**. The
file distributed under that name in the environment used for this work was
English Penn Treebank, not Chinese text, and a gazetteer mined from Chinese
annotations does not transfer to it. Do not add it to the reported results
without first confirming the corpus language and language of the voters.

## Leakage boundary

The gazetteer and suffix rules are mined from `msra_train.txt` and
`ner1998_train.csv` respectively, and evaluated on `msra_test.txt` and
`ner1998_dev.csv`. No evaluation-split text enters the rules. If you re-mine,
keep that boundary: mining on the evaluation file would inflate every number.
