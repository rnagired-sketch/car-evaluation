"""Backend of the k-nearest-neighbor application for the AME 505 Group 12 car-evaluation project.

It is independent of the decision tree packages (nothing is imported from them) and mirrors
their layout so the two models can be compared on identical splits and scores.

Modules, in pipeline order:

- `features`    ordinal attributes, the price and comfort concept scores, input sets
- `domain`      identifier types, target classes, a single `Car`
- `dataset`     validation/cleaning/encoding and stratified splits (same as the tree's)
- `metrics`     proposal-specific scores and the cross-validation scheme (same as the tree's)
- `neighbors`   the k-NN model: distance metric, K, voting rule, scaling, and the grid search
- `evaluator`   the fitted model: ratings, vote shares, the nearest training cars
- `diagnostics` K curve, learning curve, confusion matrix, prediction cost
- `stability`   re-splitting study on the decision tree's seeds
- `logs`        numbered, timed progress messages (records only; no handlers)
- `pipeline`    orchestration of all of the above

This package does no file or console I/O; see `knn_eval_io` for that.
"""

__version__ = "0.1.0"
