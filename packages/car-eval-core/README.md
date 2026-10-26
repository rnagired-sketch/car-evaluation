# car-eval-core

Backend logic for the AME 505 Group 12 car-evaluation project. Pure computation: it never
reads files, prints, or plots. `car-eval-io` does that and calls into this package.

| Module | Responsibility |
| --- | --- |
| `features` | Ordinal attributes, the expert model's *price* and *comfort* concept scores, decision-path conditions |
| `domain` | `Seed` / `CcpAlpha` newtypes, the `Acceptability` classes, a validated `Car` |
| `dataset` | Validation, cleaning and encoding (`CarDataset.from_frame`); stratified 50/20/30 splits |
| `metrics` | Accuracy, balanced accuracy, macro-F1, false acceptance/rejection, severe errors; Gini; CV scheme |
| `tree` | Tree building (grid search, repeated stratified k-fold, macro-F1) and cost-complexity pruning on validation |
| `benchmarks` | Majority baseline, the rev0 recipe, the k-NN fallback, and the adequacy check |
| `evaluator` | `CarEvaluator`: ratings, Laplace-smoothed confidence, condensed decision paths, leaf purity, importances, refit on new labels |
| `diagnostics` | Depth validation curve, learning curve, confusion matrix |
| `stability` | Re-split the data many times to measure tree instability against rev0 |
| `pipeline` | `PipelineSettings` and `run_pipeline`, which runs all of the above |
| `logs` | `Stage` and `logged_stage`: numbered, timed progress messages. The backend only emits log records; `car-eval-io` decides where they go |

```python
from car_eval_core.dataset import CarDataset
from car_eval_core.domain import Car
from car_eval_core.pipeline import PipelineSettings, run_pipeline

dataset, report = CarDataset.from_frame(raw_strings)  # columns: buying ... safety, class
result = run_pipeline(dataset, PipelineSettings())
result.evaluator.explain(Car("low", "low", "4", "4", "big", "high"))
```

Tests: `uv run pytest packages/car-eval-core` from the repository root.
