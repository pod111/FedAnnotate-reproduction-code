# FedAnnotate reproduction package

Reproduction code for the experiments in **Sec. 4.2 and Sec. 4.3** of the
FedAnnotate paper (federated intelligent annotation under a party-level
differential privacy budget).

Everything runs on CPU. No GPU, no network access, no pre-trained model.

## What is reproduced

| Paper item | Entry point | Output |
|---|---|---|
| Table 2, Fig. 4(a)(b) — privacy-utility trade-off | `run_tradeoff` | `_tradeoff.json` |
| Table 3 — five methods compared | `run_baselines` | `_baselines.json` |
| Fig. 4 | `make_fig4` | `figures/fig4_chinese_eps_tradeoff.{png,jpeg,tiff}` |
| Labeling-function mining | `mine_rules` | `_mined_fed.json`, `_mined_ner1998.json` |

## Anonymous mirror

An anonymised mirror of this repository, used for double-blind review, is
available at:

    https://anonymous.4open.science/r/FedAnnotate-reproduction-code-E537/

The GitHub repository above is the primary copy; the mirror is a read-only
anonymous view of the same commit.

## Quick start

```bash
pip install -r requirements.txt

# optional: re-derive the mined rules (both are bundled, so this is only a check)
python -m fedannotate.mine_rules
python -m fedannotate.mine_rules --dataset ner1998

# Table 2 and Figure 4(a)(b)   (~7 min)
python -m fedannotate.run_tradeoff

# Table 3                     (~9 min)
python -m fedannotate.run_baselines

# Figure 4                    (seconds)
python -m fedannotate.make_fig4
```

No command-line arguments are required; the corpora, the split seeds and the
noise seeds are the ones used for the paper.

## Protocol

| Setting | Value |
|---|---|
| Parties | K = 3 |
| Label skew | Dirichlet concentration alpha in {1.0, 0.3} |
| Privacy budget | epsilon from 64 down to 2, party-level Gaussian mechanism |
| Mechanism sensitivity | 0.35 on accuracies, 0.0875 on class priors |
| Repetitions | 2 split seeds x 5 noise seeds, mean and standard deviation |
| Communication | 19 sufficient statistics per round (8 voter accuracies, 4 class priors, plus metadata) |

Voting is done by eight heuristic labeling functions. The gazetteer and the
two-character suffix rules are mined from each corpus's own annotation text
(MSRA-NER: 1 172 gazetteer entries at purity >= 0.97, 377 suffix rules at
>= 0.90; government-news corpus: 58 entries at >= 0.90, 16 rules at >= 0.85), so
no rule carries evaluation-split information.

## Corpora

The corpora are **not** redistributed here. See `data/README.md` for how to
obtain each one and where to place it.

| File | Role |
|---|---|
| `msra_train.txt` | mining source for the MSRA-NER rules |
| `msra_test.txt` | evaluation, 6 181 spans |
| `ner1998_train.csv` | mining source for the government-news rules |
| `ner1998_dev.csv` | evaluation, 194 spans |

Two things about these files are worth stating plainly, because both were
sources of wrong numbers before they were caught:

- **CoNLL-PPP is English.** An earlier draft of the paper described a
  "Chinese sports and news" corpus with that name and reported a macro-F1 of
  0.13. The file actually obtained was English Penn Treebank, and the Chinese
  gazetteer simply does not apply to it. The reader is still in `core.py` so
  the negative result stays reproducible, but Sec. 4.3 does not report it.
- **`ner1998_dev.csv` and `ner1998_test.csv` are the same 100 documents.** Use
  exactly one. The evaluation file is `ner1998_dev.csv`.

## Implementation notes

These are the decisions that a re-implementer is most likely to get wrong. Each
one produced a wrong number before it was fixed, so each is marked in the code
next to the relevant line.

**The Gaussian noise must be applied to the posterior, not only to the
aggregated weights.** The argmax is insensitive to an equal-magnitude
perturbation of the weights, so an epsilon sweep that noises only the weights
returns identical numbers at every setting. `core.predict_with` therefore adds
the noise after normalising.

**The noise must reach the local estimates too.** Eq. (3) compensates a local
accuracy estimate that the privacy mechanism has degraded toward 0.5. If the
noise is added only to the aggregate, the local estimates stay clean, the gap
they should be corrected for is about 0.07, the compensation coefficient
collapses, and the mechanism does nothing. `run_tradeoff.one_run` noises both.

**The compensation must be modulated by the size of the gap.** Adding a flat
bonus to every voter destroys the weight ordering and yields a negative gain.
`core.semantic_compensation` uses `a' = a + lambda (a_ref - a)` with
`lambda = |gap| / (|gap| + tau)`, which leaves well-estimated voters alone.

**Drop high-coverage, low-precision voters.** Rules of the form "any surface
of length k is class c" reach 70-80% coverage at 0-15% precision and swamp
everything else. Removing them moved macro-F1 from 0.116 to 0.52. The
`noisy_len` voter kept in this package is intentional but is the weakest one;
it is reported as part of the voter set, not hidden.

**Probe sets must be sampled at random, not taken from the tail.** These
corpora are concatenations of several sources, so the tail is not
representative. A tail probe set pushed the membership-inference AUC below 0.5,
which reads as the attack helping the adversary.

**Small corpora need a floor on the party sizes.** With 194 spans, Dirichlet
can starve a party; `partition` tops every party up so all K parties stay
trainable. The same applies to the probe hold-out: a fixed minimum that suits a
6 000-span corpus will consume the whole of a 194-span one.

**A membership-inference AUC near 0.5 is not by itself evidence of privacy.**
It says only that this attack found no signal against this message structure.
The released object is 19 sufficient statistics, and only party-level noise was
applied; a sample-level attack was not run. Sec. 4.3 states this limitation.

## Expected output

```
MSRA-NER  n=6181 spans
  alpha=1.0 eps=64: 0.586 ± 0.014
  ...
News-NER  n=194 spans
  alpha=1.0 eps=64: 0.378 ± 0.015
  ...
```

Compare against the bundled `_tradeoff.json` and `_baselines.json`.
Run-to-run variation should be nil, because every seed is fixed; if you see
different numbers, a seed or a corpus file has changed.

## License

MIT. See `LICENSE`.