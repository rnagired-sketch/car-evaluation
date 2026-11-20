# knn-eval-core

Backend of the k-NN application. Pure computation: it never reads files, prints or plots;
`knn-eval-io` does that and calls into this package. It does not depend on the decision tree packages.

| Module | Responsibility |
| --- | --- |
| `features` | Ordinal attributes, the *price* and *comfort* concept scores, `InputSet` (raw or with concepts) |
| `domain` | `Seed` newtype, the `Acceptability` classes, a validated `Car` |
| `dataset` | Validation, cleaning, encoding and the stratified 50/20/30 split (identical to the tree's) |
| `metrics` | Accuracy, balanced accuracy, macro-F1, false acceptance/rejection, severe errors; CV scheme |
| `neighbors` | `DistanceMetric`, `Voting` (majority, 1/d, 1/d²), `Scaling`, `KnnParams`, and the grid search |
| `evaluator` | `KnnEvaluator`: ratings, vote shares, the K nearest training cars, importances, new labels |
| `diagnostics` | Choosing-K curve, learning curve, confusion matrix, prediction cost |
| `stability` | Re-split study on the decision tree's seeds |
| `logs` | Numbered, timed progress messages (records only) |
| `pipeline` | `PipelineSettings` and `run_pipeline`, which runs all of the above |

Tests: `uv run pytest packages/knn-eval-core` from the `knn` folder.
