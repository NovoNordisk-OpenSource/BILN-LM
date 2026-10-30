# BILN-LM

A masked language model over peptides written in **BILN** (from publication [BILN – A Human-readable Line Notation for Complex Peptides](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00703), 
Journal of Chemical Information and Modelling, 2022), and a small API to turn BILN strings into embeddings.


## Install

We recommend to use a dedicated virtual environment:

```bash
cd BILN-LM
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/python -m pip install -e .
```

`requirements-optional.txt` adds `rdkit`/`molfeat`
(SMILES and fingerprint baselines) and `optuna` (hyper-parameter search). None of
them are needed for parsing, pretraining or embedding.

## Quickstart

```bash
# 1. synthetic corpus for pretraining (optional)
python scripts/generate_synthetic.py --config synthetic.yaml

# 2. pretrain example if synthetic data was generated
python scripts/pretrain.py --config pretrain.yaml \
    data.files=[biln_db.csv,synthetic_biln.csv] paths.output_dir=outputs/pretrain

# 3. embeddings
python scripts/embed.py --input data/biln_db.csv --column biln \
    --model-dir outputs/pretrain --out embeddings.npy
```

From Python:

```python
from biln_lm import embed_biln

vectors = embed_biln(
    ["C-P-W-H-L-L-P-F-C", "ac-dPhe(4-Cl)(1,3)-Me_dR-G-am"],
    model_dir="outputs/pretrain",
    pooling="mean",
)   # -> np.ndarray of shape (2, hidden_size)
```

Use `BilnEmbedder` instead when embedding many batches, so the model is loaded once:

```python
from biln_lm import BilnEmbedder

embedder = BilnEmbedder("outputs/pretrain", cache_dir=".cache/embeddings")
vectors = embedder.encode(sequences, batch_size=64, show_progress=True)
```

## Configuration

Every entrypoint takes `--config <file>` plus any number of `key=value` overrides,
and `--print-config` shows the merged result without running anything:

```bash
python scripts/pretrain.py --config pretrain.yaml \
    model.num_hidden_layers=8 training.learning_rate=3e-4 data.max_samples=5000
```

Configs merge in the order `default.yaml` → the file's `_base_` chain → the file
itself → CLI overrides.

| Key | Meaning |
| --- | --- |
| `paths.data_dir` | where corpus and benchmark CSVs live |
| `paths.output_dir` | where the run writes its artefacts |
| `tokenizer.kind` | `monomer` (default) or `bpe` |
| `tokenizer.path` | reuse an existing tokenizer instead of building one |
| `data.files` / `data.column` | corpus files and the BILN column |
| `data.max_length` | tokens per sequence |
| `model.pretrained_path` | continue pretraining from a directory or hub id |
| `model.*` | ESM architecture (hidden size, layers, heads, position embeddings) |
| `training.*` | forwarded to `transformers.TrainingArguments` |

## Pretraining

`scripts/pretrain.py` writes a standard `save_pretrained` directory, so the result
loads with plain `AutoModel.from_pretrained(...)` as well as with
`biln_lm.load_pretrained(...)`. Alongside the weights it saves the resolved
`config.yaml`, `training_history.csv` and `summary.json`.

Continue pretraining an existing checkpoint on new data:

```bash
python scripts/pretrain.py --config pretrain.yaml \
    model.pretrained_path=outputs/pretrain \
    tokenizer.path=outputs/pretrain \
    data.files=[my_new_corpus.csv] paths.output_dir=outputs/pretrain_v2
```

Optional hyper-parameter search (needs `optuna`):

```bash
python scripts/hpo.py --config hpo.yaml hpo.n_trials=30
```

Every trial starts from the config as given on the command line and merges the
sampled values on top, so `data.max_samples=5000` on the CLI applies to all trials.

## Tokenization

BILN is `chain ("." chain)*`, each chain `monomer ("-" monomer)*`, each monomer a
symbol plus optional `(bond,rgroup)` annotations. Splitting is not a plain
`split("-")`: symbols may contain hyphens inside parentheses (`Phe(4-Cl)`) or even
start with one (`-pip`), and `biln_lm.biln` handles both.

The default **monomer** tokenizer emits one token per residue, drawn from a 3,000+
symbol library, with the bond annotations as separate tokens:

```
"ac-dPhe(4-Cl)(1,3)-Me_dR-G-am"  ->  ['ac', 'dPhe(4-Cl)', '(1,3)', 'Me_dR', 'G', 'am']
```

Across the bundled corpora (~26k sequences) this gives a 0% `[UNK]` rate. The
 **BPE** tokenizer remains available via `tokenizer.kind=bpe`.

## Synthetic data

`scripts/generate_synthetic.py` produces two kinds of sequences and writes them to
one CSV with a `source` column:

- **de-novo** — residues sampled from the monomer library, with configurable
  length, non-canonical fraction, terminal caps, head-to-tail macrocycles,
  disulfide bridges and multi-chain assemblies.
- **augmented** — point mutations, natural-analog swaps, truncations and segment
  shuffles of an existing corpus. Residues carrying bonds are never altered, so
  the topology stays consistent.

Everything is filtered through `is_valid_biln` and deduplicated.

## Downstream benchmark

```bash
python scripts/evaluate_downstream.py --config downstream.yaml \
    downstream.model_dir=outputs/pretrain 'downstream.tasks=[c-cpp,nc-cpp]'
```

Peptides are embedded with the pretrained model, optionally concatenated with ESM-2
protein embeddings for the binding tasks, and passed to SVM / random forest / KNN
heads. Results land in `paths.output_dir/results.csv`.

The binding datasets write residues as three-letter codes (`HIS-PRO-ARG`);
`downstream.normalize_biln` maps those onto library symbols before tokenization.

## Tests

There is a set of modules to test all the packages functionalities. This include public and toy sequences:

```bash
python -m pytest tests -q
```

## References

If you use BILN-LM in your work, please cite the following papers:

* [Placeholder BILN-LM application note]() , XXX, 202X.
* [BILN – A Human-readable Line Notation for Complex Peptides](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00703), Journal of Chemical Information and Modelling, 2022.
