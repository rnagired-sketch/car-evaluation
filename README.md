# Car Evaluation: a robust decision tree (AME 505, Group 12)

Classifies cars from the [UCI Car Evaluation dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)
as `unacc`, `acc`, `good` or `vgood`, utilizing CART tree building, cost-complexity pruning, Gini purity, and a k-NN fallback.

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
└── .github/workflows/ci.yml  lint, type-check and test on Python 3.11 to 3.13
```

The dependency only points one way: `car-eval-io` depends on `car-eval-core`.
The core package holds frozen dataclasses and pure functions. The io package turns YAML into those
dataclasses with [cattrs](https://catt.rs), so each class's own validation (`__post_init__`) also
checks every value read from a config file.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rnagired-sketch/car-evaluation.git && cd car-evaluation
```

```bash
uv sync --all-packages
```

```bash
uv run car-eval run --config configs/config.yaml --data data/car.data --output outputs
```

```bash
uv run car-eval explain --config configs/config.yaml --output outputs
```

```bash
uv run car-eval show-config --config configs/config.yaml
```


| Command | What it does |
| --- | --- |
| `run` | Validate and split the data, build and prune a tree for each feature set, pick one on the validation split, compare it with the baseline / rev0 / k-NN, score it once on the test split, run the stability study, explain `cars`, and write the outputs |
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
Relative paths are resolved against the current working directory.

Outputs, written to the `--output` folder:

- `report.txt`: the console report.
- `results.json`: every metric, in machine-readable form.
- `model.json`: a readable recipe that rebuilds the trained tree deterministically.
- `figures/*.png`: class balance, depth curve, pruning path, learning curve, confusion matrix, feature importance, and the tree.

## Configuration

Everything is in [`configs/config.yaml`](configs/config.yaml), with every key commented. The file
has one set of base settings (the full evaluation plan) and a `profiles:` section listing only
what each variant changes:

| Profile | Use |
| --- | --- |
| `default` | The base settings: full grid, 3× repeated 5-fold CV, 20-split stability study, figures saved |
| `quick` | Smoke run in well under a minute: small grid, one CV repeat, 3-split study, no figures |
| `colab` | Figures shown inline as well as saved (the notebook passes the Drive locations) |

Select one with the `profile:` key at the top of the file. Nested sections merge key by key,
and lists and values replace. `show-config` prints the merged result. Mistakes are reported
with their location, and every problem is listed:

```
invalid configuration configs/mine.yaml:
  - unknown key(s): sed @ $.pipeline
  - buying='cheap'; expected one of ('low', 'med', 'high', 'vhigh') @ $.cars[0]
```

Exit codes: `0` success, `1` runtime failure (for example, no `model.json` yet), `2` invalid
configuration or command-line arguments.

## Method and findings

| Proposal / lecture item | Where |
| --- | --- |
| Validate, clean and encode the attributes | `car_eval_core.dataset.CarDataset.from_frame` |
| 50 / 20 / 30 stratified split | `DataSplits.stratified` |
| Class imbalance: macro-F1 model selection, class weighting in the search | `car_eval_core.tree.tune_tree` |
| Tree building, then tree pruning (Lecture 3B) | `tune_tree`, then `prune_on_validation` |
| Gini purity of the leaves | `CarEvaluator.purity` |
| Unacceptable rated acceptable, and the reverse | `Scores.false_acceptance`, `Scores.false_rejection` |
| k-NN fallback and adequacy check | `car_eval_core.benchmarks` |
| Instability of trees | `car_eval_core.stability` |
| Decision path with a rating and a confidence | `CarEvaluator.explain` |
| Monitor and update with new labels | `CarEvaluator.with_new_labels` |

Main finding: the labels come from an expert hierarchy (Bohanec & Rajkovič, 1988) in which buying +
maint form *price* and doors + persons + lug_boot form *comfort*. Adding those two rank sums as
features lets the tree split on the combinations directly. Across 20 random splits, test macro-F1
rises from 0.930 ± 0.022 (rev0) to 0.979 ± 0.017, the tree needs about half as many leaves, and
rev1 matches or beats rev0 on every split. Pruning and macro-F1 scoring alone do not change the tree,
because the dataset has no label noise to prune away.

## Development

```bash
uv run pytest
```

```bash
uv run pytest -m "not slow"
```

```bash
uv run ruff check && uv run ruff format --check && uv run pyright
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
