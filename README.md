# Car Evaluation: a robust decision tree (AME 505, Group 12)

Classifies cars from the [UCI Car Evaluation dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)
as `unacc`, `acc`, `good` or `vgood` with one CART decision tree. The tree reads the six car
attributes plus two concept scores, *price* and *comfort*, and is grown by grid search, pruned by
cost complexity and checked for Gini purity. On the held-out test split it rates 518 of 519 cars
correctly (macro-F1 0.999); see [The model](#the-model).

## Repository layout

```
car-evaluation/
├── pyproject.toml            uv workspace root: shared lockfile and dev tooling (ruff, pyright, pytest)
├── uv.lock
├── configs/
│   └── config.yaml           what a run does: base settings plus the quick and colab profiles
├── data/                     car.data (UCI, CC BY 4.0) so runs and tests work offline
├── notebooks/                Colab notebook: installs the packages, runs the colab profile
├── packages/
│   ├── car-eval-core/        backend: all modelling logic, no file or console I/O
│   │   ├── src/car_eval_core/
│   │   └── tests/
│   └── car-eval-io/          application layer: YAML/JSON parsing, cattrs object construction,
│       ├── src/car_eval_io/  reports, plots and the argparse CLI (__main__.py)
│       └── tests/
├── knn/                      separate k-NN application for comparison (own workspace and config)
└── .github/workflows/ci.yml  lint, type-check and test both apps on Python 3.11 to 3.13
```

The dependency only points one way: `car-eval-io` depends on `car-eval-core`.
The core package holds frozen dataclasses and pure functions. The io package turns YAML into those
dataclasses with [cattrs](https://catt.rs), so each class's own validation (`__post_init__`) also
checks every value read from a config file.

## Getting started (VS Code, Windows PowerShell or macOS Terminal)

The commands below are the same in Windows PowerShell and the macOS Terminal unless a step says
otherwise. Forward slashes in paths work on Windows too.

### 1. Install the prerequisites

You need Git, VS Code and uv. Python itself is optional: uv downloads Python 3.12 if it isn't installed.

| Tool | Windows (PowerShell) | macOS (Terminal) |
| --- | --- | --- |
| Git | `winget install --id Git.Git -e` | `xcode-select --install` |
| VS Code | `winget install --id Microsoft.VisualStudioCode -e` | Download from [code.visualstudio.com](https://code.visualstudio.com), or `brew install --cask visual-studio-code` |
| uv | `winget install --id astral-sh.uv -e` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

After installing:

1. Close and reopen the terminal so it picks up the new `PATH`.
2. Check each tool: `git --version`, `code --version`, `uv --version`.
3. In VS Code, open the Extensions view (`Ctrl+Shift+X` on Windows, `Cmd+Shift+X` on Mac) and
   install **Python** by Microsoft.

On a Mac, if `code` is not found, open VS Code, press `Cmd+Shift+P` and run
**Shell Command: Install 'code' command in PATH**. If `uv` is not found, see
[Troubleshooting](#troubleshooting). Other install options are in the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Clone the repository and open it in VS Code

```bash
git clone https://github.com/rnagired-sketch/car-evaluation.git
```

```bash
code car-evaluation
```

Or use **File > Open Folder...** and choose `car-evaluation`. Open the repository root, never one
of the `packages` subfolders: uv and VS Code read the root `pyproject.toml` to find both packages.
When VS Code asks whether you trust the authors, choose **Yes, I trust the authors**.

Open the integrated terminal with **Terminal > New Terminal** (`` Ctrl+` `` on both systems). It
starts in the repository root; run every command below from there. If you already have a clone,
update it first with `git pull origin main`.

### 3. Set up the environment

```bash
uv sync --all-packages
```

This creates `.venv` with Python and every dependency, pinned by `uv.lock`. Run it again after a
`git pull` that changes `uv.lock` or a `pyproject.toml`. You never need to activate `.venv`;
`uv run` uses it automatically.

Then point VS Code at it: press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac), run
**Python: Select Interpreter** and choose the `.venv` entry. If it isn't listed, choose
**Enter interpreter path...** and give `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Mac).

### 4. Run the application

Run the full evaluation (under a minute):

```bash
uv run car-eval run --config configs/config.yaml --data data/car.data --output outputs
```

Rate and explain the cars listed in the config, using the model the run saved:

```bash
uv run car-eval explain --config configs/config.yaml --output outputs
```

Print the settings in effect, with every default filled in:

```bash
uv run car-eval show-config --config configs/config.yaml
```

`uv run python -m car_eval_io ...` is equivalent. Add `--help` after any command to list its options.
To try a faster run first, set `profile: quick` at the top of `configs/config.yaml` (see
[Configuration](#configuration)).

| Command | What it does |
| --- | --- |
| `run` | Validate and split the data, build the tree (grid search) and prune it on the validation split, score it once on the test split, run the stability study, explain `cars`, and write the outputs |
| `explain` | Rebuild the model from `model.json` and rate and explain the config's `cars` |
| `show-config` | Print the resolved configuration with every default filled in |

| Option | Commands | Meaning |
| --- | --- | --- |
| `-c`, `--config FILE` | all (required) | The YAML configuration file |
| `-d`, `--data FILE` | `run`, `explain` | A local `car.data` file |
| `--download` | `run`, `explain` | Download `car.data` from the UCI repository instead |
| `-o`, `--output DIR` | `run`, `explain` | Results folder (default `outputs`); `explain` reads `model.json` from it |

`run` needs exactly one of `--data` or `--download`. Without either, `explain` re-reads the
source recorded in `model.json` and refuses data whose size differs from the training data.
Relative paths are resolved against the folder the terminal is in, so run the commands from the
repository root. The output folder is created as soon as `run` starts, and the first log lines
print its full path (`results folder: /.../car-evaluation/outputs`); the last ones list every file
written.

Outputs, written to the `--output` folder:

- `report.txt`: the console report.
- `results.json`: every metric, in machine-readable form.
- `model.json`: a readable recipe that rebuilds the trained tree deterministically.
- `figures/*.png`: class balance, depth curve, pruning path, learning curve, confusion matrix, feature importance, and the tree.
- `car-eval.log`: every progress and error message, appended by each `run` and `explain`.

Open them from the VS Code Explorer. `outputs/` is ignored by Git.

### Progress and error messages

While a command runs, it logs what it is doing to the terminal: numbered stages with their
timings, the scores as they are computed, and every warning and error.

```
20:01:56 INFO    car_eval_core.pipeline: [2/5] building and pruning the tree on 6 raw attributes + price & comfort scores ...
20:02:20 INFO    car_eval_core.tree: best CV macro-F1 0.962 +/- 0.044 with entropy, max_depth 12, ...
20:02:21 INFO    car_eval_core.pipeline: [2/5] building and pruning the tree on 6 raw attributes + price & comfort scores done in 24.4 s
20:02:26 INFO    car_eval_core.stability:   split 4/20 (seed 45): test macro-F1 0.985, 33 leaves
21:01:18 ERROR   car_eval_io: cannot read model recipe outputs/model.json: No such file or directory. Run `car-eval run` first.
```

Log messages go to standard error and the report to standard output, so
`uv run car-eval run ... > report.txt` saves the report alone while progress stays on screen.
Set the amount of detail under `logging:` in `configs/config.yaml`: `level` is `debug`, `info`
(default), `warning` or `error`, and `save: false` turns off `car-eval.log`. An unexpected error
is logged with its full traceback.

### Troubleshooting

| Problem | Fix |
| --- | --- |
| `uv` is not recognized (Windows) or `command not found: uv` (Mac) | Close and reopen the terminal, and restart VS Code if you use its terminal. On a Mac, `source $HOME/.local/bin/env` fixes the current window; if new windows still fail, run `echo 'source $HOME/.local/bin/env' >> ~/.zshrc` once. |
| `car-eval` not found, or `ModuleNotFoundError: car_eval_core` | Run `uv sync --all-packages` from the repository root. |
| VS Code underlines the `car_eval_core` / `car_eval_io` imports | Select the `.venv` interpreter (step 3). |
| `one of the arguments -d/--data --download is required` | `run` needs `--data data/car.data` or `--download`. |
| `no such file: data/car.data` | The terminal is not in the repository root: `cd car-evaluation`, or give the full path. |
| `cannot read model recipe ... Run car-eval run first` | `explain` reads `model.json` from `--output`; run `run` first with the same `--output`. |
| `invalid configuration ...` | Each line names a bad key and where it is (for example `@ $.pipeline`); fix it in `configs/config.yaml`. |
| `uv sync` fails with *Access is denied* (Windows) | Close other terminals or programs using `.venv` and rerun. Keeping the repository outside OneDrive avoids sync locks. |
| No figures appear | They are saved to `outputs/figures`, not shown. The `quick` profile saves none. |

## Configuration

Everything is in [`configs/config.yaml`](configs/config.yaml), with every key commented. The file
has one set of base settings (the full evaluation plan) and a `profiles:` section listing only
what each variant changes:

| Profile | Use |
| --- | --- |
| `default` | The base settings: full grid, 3× repeated 5-fold CV, 20-split stability study, figures saved |
| `quick` | Smoke run in about 10 seconds: small grid, one CV repeat, 3-split study, no figures |
| `colab` | Figures shown inline as well as saved (the notebook passes the Drive locations) |

Select one with the `profile:` key at the top of the file. Nested sections merge key by key,
and lists and values replace. `show-config` prints the merged result. Mistakes are reported
with their location, and every problem is listed:

```
invalid configuration configs/mine.yaml:
  - unknown key(s): sed @ $.pipeline
  - buying='cheap'; expected one of ('low', 'med', 'high', 'vhigh') @ $.cars[0]
```

Exit codes: `0` success, `1` runtime failure (for example, no `model.json` yet), `130`
interrupted with `Ctrl+C`, `2` invalid
configuration or command-line arguments.

## The model

One CART decision tree rates each car. With the default configuration (seed 42) it scores
macro-F1 0.999 on the untouched test split, and 0.979 ± 0.017 across 20 random re-splits of the data.

### Inputs

Each attribute is encoded as its rank, lowest level first. Two concept scores are added from the
expert model the labels came from (Bohanec & Rajkovič, 1988), where buying price and maintenance
cost form PRICE, and doors, seats and boot size form COMFORT. With them the tree can split on those
combinations directly instead of rebuilding them from many single-attribute splits.

| Input | Levels | Encoded as |
| --- | --- | --- |
| `buying`, `maint` | low, med, high, vhigh | 0–3 |
| `doors` | 2, 3, 4, 5more | 0–3 |
| `persons` | 2, 4, more | 0–2 |
| `lug_boot` | small, med, big | 0–2 |
| `safety` | low, med, high | 0–2 |
| `price` | buying + maint | 0–6 (0 = cheapest) |
| `comfort` | doors + persons + lug_boot | 0–7 (0 = least roomy) |

### How it is trained

1. Validate, clean and encode the 1728 rows (none are dropped from the UCI file).
2. Split them 50 / 20 / 30 into training (863), validation (346) and test (519) cars, stratified
   so each split keeps the class mix (70% `unacc`, 22% `acc`, 4% `good`, 4% `vgood`).
3. **Build:** grid-search 120 hyperparameter combinations (criterion, maximum depth, minimum
   samples per split and per leaf, class weighting) with 3× repeated 5-fold cross-validation on the
   training cars. Candidates are ranked by macro-F1, so the rare `good` and `vgood` classes count
   as much as `unacc`.
4. **Prune:** score every tree on the minimal cost-complexity pruning path on the validation cars
   and keep the smallest one with the best validation macro-F1.
5. **Test:** fit that tree on the training cars and score it once on the test cars.

### The trained tree

| Setting | Value |
| --- | --- |
| Split criterion | entropy |
| Maximum depth allowed / reached | 12 / 11 |
| Minimum samples per split / per leaf | 2 / 1 |
| Class weighting | none |
| Pruning strength (`ccp_alpha`) | 0: pruning removes no branch, because every leaf is already pure |
| Leaves | 35 |
| Cross-validated macro-F1 (training) | 0.962 ± 0.044 |
| Validation macro-F1 | 0.996 |

Gini impurity falls from 0.456 at the root to 0.000 averaged over the leaves. The root split is
on `safety`: cars with low safety are always unacceptable.

### Results on the test split

| Class | Precision | Recall | F1 | Cars |
| --- | --- | --- | --- | --- |
| `unacc` | 0.997 | 1.000 | 0.999 | 363 |
| `acc` | 1.000 | 0.991 | 0.996 | 115 |
| `good` | 1.000 | 1.000 | 1.000 | 21 |
| `vgood` | 1.000 | 1.000 | 1.000 | 20 |

Accuracy is 0.998 and macro-F1 0.999. No unacceptable car is rated acceptable (false acceptance
0.000), and 1 of 156 acceptable cars is rated unacceptable (false rejection 0.006). No rating is
off by two or more classes. The one error is `buying=med, maint=med, doors=5more, persons=4,
lug_boot=small, safety=med`: truly `acc`, rated `unacc` with confidence 0.81 from a leaf of 12
unacceptable training cars.

Shuffling one attribute on the test cars lowers macro-F1 by 0.470 for `safety`, 0.419 for
`buying`, 0.342 for `maint`, 0.322 for `persons`, 0.262 for `lug_boot` and 0.121 for `doors`.

### Stability

Decision trees can change shape when the training data changes slightly (Lecture 3B). Refitting
the same settings on 20 random 50 / 20 / 30 splits gives test macro-F1 0.979 ± 0.017 (lowest
0.951), a false-acceptance rate of 0.004 on average, 33 ± 2 leaves, and a root split on `safety`
14 times and on `persons` 6 times.

These numbers come from `uv run car-eval run --config configs/config.yaml --data data/car.data`;
the run is deterministic, so it reproduces them.

### Where each step lives in the code

| Proposal / lecture item | Where |
| --- | --- |
| Validate, clean and encode the attributes | `car_eval_core.dataset.CarDataset.from_frame` |
| Price and comfort concept scores | `car_eval_core.features.model_inputs` |
| 50 / 20 / 30 stratified split | `DataSplits.stratified` |
| Class imbalance: macro-F1 model selection, class weighting in the search | `car_eval_core.tree.tune_tree` |
| Tree building, then tree pruning (Lecture 3B) | `tune_tree`, then `prune_on_validation` |
| Gini purity of the leaves | `CarEvaluator.purity` |
| Unacceptable rated acceptable, and the reverse | `Scores.false_acceptance`, `Scores.false_rejection` |
| Instability of trees | `car_eval_core.stability` |
| Decision path with a rating and a confidence | `CarEvaluator.explain` |
| Monitor and update with new labels | `CarEvaluator.with_new_labels` |

## k-NN comparison model

[`knn/`](knn/README.md) holds a separate k-nearest-neighbor application for comparison with this
tree. It mirrors this folder structure but has its own workspace, lockfile, config
(`knn/configs/knn.yaml`) and command (`knn-eval`), and the two apps never import each other. It
uses the same data, seed, split and scores. See [knn/README.md](knn/README.md) to run it and for
the side-by-side results.

## Development

```bash
uv run pytest
```

```bash
uv run pytest -m "not slow"
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

Code style follows [Writing Python like it's Rust](https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html):

## Google Colab

Open `notebooks/car_evaluation_colab.ipynb` and run all cells. The notebook
pip-installs both packages, mounts Drive and runs `configs/config.yaml` with the `colab` profile,
showing figures inline. The data and output locations on Drive are set in the notebook's run cell,
the same way `--data` and `--output` are on the command line.

## Data

See [data/README.md](data/README.md). Car Evaluation dataset © M. Bohanec, CC BY 4.0.

## References

### Project and course material
- Miranda, E., Hanna, P., & Nagireddy, R. (2026). *Project Proposal: Car Evaluation* (Group 12). AME 505: Machine Learning for Engineering Applications, University of Southern California.
- Jin, Y. (2026). *Lecture 3A: Introduction to Machine Learning and Classification*. AME 505, University of Southern California.
- Jin, Y. (2026). *Lecture 3B: K-Nearest Neighbor and Decision Tree*. AME 505, University of Southern California.
- Jin, Y. (2026). *Machine Learning for Engineering Applications*. Course textbook.

### Dataset and domain model
- Bohanec, M. (1988). *Car Evaluation* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48 (CC BY 4.0)
- Bohanec, M., & Rajkovič, V. (1988). Knowledge acquisition and explanation for multi-attribute decision making. *8th International Workshop on Expert Systems and their Applications*, Avignon, France, 59–78. The source of the PRICE and COMFORT concept hierarchy behind the `price` and `comfort` features.

### Methods
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth. CART tree building and minimal cost-complexity pruning.

### Software
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
- McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. (pandas)
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
- cattrs: https://catt.rs
- PyYAML: https://pyyaml.org
- uv: https://docs.astral.sh/uv/
- pytest, Ruff and Pyright, for testing, linting and type checking.

### Code style
- Beránek, J. ("Kobzol") (2023, May 20). *Writing Python like it's Rust*. https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html

### AI assistance
- All code structured, revised, and acceptance tested by Rupin Nagireddy, Eduardo Miranda, and Pierre Hanna.
- Supporting code and documentation were developed with the help of Claude (Anthropic) through Claude Code.
