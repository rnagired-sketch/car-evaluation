"""Backend for the AME 505 Group 12 car-evaluation project.

Modules, in pipeline order:

- `features`    ordinal attributes, concept scores, decision-path conditions
- `domain`      identifier types, target classes, a single `Car`
- `dataset`     validation/cleaning/encoding and stratified splits
- `metrics`     proposal-specific scores, Gini impurity, cross-validation scheme
- `tree`        CART tree building (grid search) and pruning (cost complexity)
- `benchmarks`  majority baseline, rev0 recipe, k-NN fallback, adequacy check
- `evaluator`   the trained model: ratings, confidence, decision paths, purity
- `diagnostics` depth/learning curves and the confusion matrix
- `stability`   re-splitting study for tree instability
- `pipeline`    orchestration of all of the above

This package does no file or console I/O; see `car_eval_io` for that.
"""

__version__ = "0.1.0"
