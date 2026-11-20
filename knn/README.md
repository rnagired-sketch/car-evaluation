# k-Nearest Neighbors model (comparison to the decision tree)

A k-nearest-neighbor (k-NN) classifier for the same UCI Car Evaluation data, built to be compared
with the decision tree in the repository root. It follows the k-NN material of Lecture 3B: a
stored set of training cars, a distance metric, a K value, a voting rule and feature scaling.

It is a separate application. It has its own uv workspace, lockfile, `.venv`, config file
([`configs/knn.yaml`](configs/knn.yaml)) and command (`knn-eval`). It never imports the decision
tree packages, and the decision tree never imports it; tests enforce both. It does use the same
dataset file, seed, 50/20/30 stratified split and scores as the tree, so the two models are
trained and tested on exactly the same cars.

## Folder layout

```
knn/
├── pyproject.toml            uv workspace root for the k-NN app (separate from the tree's)
├── uv.lock
├── configs/
│   └── knn.yaml              what a run does: base settings plus the quick and colab profiles
└── packages/
    ├── knn-eval-core/        backend: all model logic, no file or console I/O
    │   ├── src/knn_eval_core/
    │   └── tests/
    └── knn-eval-io/          application layer: YAML/JSON parsing, cattrs object construction,
        ├── src/knn_eval_io/  reports, plots and the argparse CLI (__main__.py)
        └── tests/
```

The data comes from the repository's shared [`data/car.data`](../data/car.data).

## Running it

Install Git, VS Code and uv as described in the [main README](../README.md#1-install-the-prerequisites).
Then work **inside the `knn` folder**, which has its own environment:

```bash
cd knn
```

```bash
uv sync --all-packages
```

```bash
uv run knn-eval run --config configs/knn.yaml --data ../data/car.data --output outputs
```

```bash
uv run knn-eval explain --config configs/knn.yaml --output outputs
```

```bash
uv run knn-eval show-config --config configs/knn.yaml
```

A full run takes about a minute; `profile: quick` at the top of `configs/knn.yaml` takes a few
seconds. The options, exit codes and logging work exactly as in the decision tree app:
`-c/--config`, `-d/--data` or `--download`, `-o/--output`. Progress and errors are logged to the
terminal and to `<output>/knn-eval.log`, and the report goes to standard output.

Outputs, written to the `--output` folder (`knn/outputs/`, ignored by Git):

- `report.txt`: the full report, also printed to the terminal.
- `results.json`: every metric in machine-readable form (`"model": "k-NN"`).
- `model.json`: the recipe that rebuilds the fitted model (used by `explain`).
- `explanations.json`: written by `explain`.
- `knn-eval.log`: every progress and error message.
- `figures/`: class balance, best score per distance metric, choosing K, learning curve,
  confusion matrix, attribute importance.

## What the model compares (Lecture 3B)

| Lecture 3B element | Setting in `configs/knn.yaml` | Options |
| --- | --- | --- |
| Distance metric | `pipeline.grid.metric` | `euclidean`, `manhattan`, `minkowski` (with `minkowski_p`), `hamming` (categorical), `cosine` |
| K value | `pipeline.grid.k` | 1, 3, 5, 7, 9, 11, 15, 21 by default |
| Voting | `pipeline.grid.voting` | `majority`, `inverse_distance` (1/d), `inverse_square` (1/d², the lecture's weight factor) |
| Feature scaling | `pipeline.grid.scaling` | `standard`, `minmax`, `none` |
| Inputs | `pipeline.inputs` | `with_concepts` (the tree's inputs, default) or `raw` |

All combinations (120 by default) are scored by macro-F1 with 3× repeated stratified 5-fold
cross-validation on the 863 training cars, all on the same folds. The best one is checked on the
346 validation cars and tested once on the 519 test cars. The report also covers the lecture's
other points: the best setting for each distance metric, the macro-F1 for every K, prediction
cost (k-NN compares each new car with every stored car), and which attributes matter. Every
rating is explained by listing its K nearest training cars with their distances and vote weights.

## Results (default configuration, seed 42)

**Chosen setting:** K = 7, cosine distance, 1/d voting, standard scaling (cross-validated
macro-F1 0.860 ± 0.045; validation 0.911).

| Distance metric | Best K | Voting | CV macro-F1 |
| --- | --- | --- | --- |
| cosine | 7 | 1/d | 0.860 ± 0.045 |
| euclidean | 5 | 1/d² | 0.829 ± 0.044 |
| minkowski (p = 3) | 7 | 1/d² | 0.815 ± 0.033 |
| manhattan | 5 | 1/d | 0.803 ± 0.055 |
| hamming | 7 | majority | 0.705 ± 0.051 |

**Test split (519 cars):**

| Class | Precision | Recall | F1 | Cars |
| --- | --- | --- | --- | --- |
| `unacc` | 0.957 | 0.986 | 0.972 | 363 |
| `acc` | 0.904 | 0.817 | 0.858 | 115 |
| `good` | 0.810 | 0.810 | 0.810 | 21 |
| `vgood` | 0.900 | 0.900 | 0.900 | 20 |

### Side by side with the decision tree

Same split, same cars, same scores; the tree's figures come from its own default run.

| Measure | Decision tree | k-NN |
| --- | --- | --- |
| Test macro-F1 | 0.999 | 0.885 |
| Test accuracy | 0.998 | 0.938 |
| False acceptance (unacc rated acceptable) | 0.000 | 0.014 |
| False rejection (acceptable rated unacc) | 0.006 | 0.103 |
| Test cars misclassified | 1 of 519 | 32 of 519 |
| Macro-F1 over 20 re-splits | 0.979 ± 0.017 | 0.887 ± 0.023 |
| Explanation of a rating | decision path (rules) | the K nearest training cars |

Lecture 3B notes that k-NN is weak on categorical attributes. Here every input is a short ordinal
scale, many cars sit at the same distance, and the class boundaries follow expert rules that a
tree can represent exactly. The best distance, cosine, compares the pattern of a car's
standardized attributes rather than their size.

## Development

From the `knn` folder:

```bash
uv run pytest
```

```bash
uv run ruff check
```

```bash
uv run ruff format --check
```

```bash
uv run pyright
```

The tests check, among other things, that the k-NN splits are identical to the decision tree's
(row-level fingerprints), that the vote shares equal the neighbors' weights, that 1/d² voting
behaves as defined (an exact match decides alone), and that neither application imports the
other.
