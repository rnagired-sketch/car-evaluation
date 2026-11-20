"""Application layer of the k-nearest-neighbor model for the AME 505 Group 12 project.

Independent of the decision tree application: it builds only on `knn_eval_core`.

- `__main__`      argparse entry point (`knn-eval` / `python -m knn_eval_io`)
- `app`           the commands: run, explain, show-config
- `config`        the YAML configuration schema and loader (configs/knn.yaml)
- `converters`    the cattrs converter that builds typed objects from YAML/JSON
- `sources`       where the raw dataset comes from and how it is read
- `paths`         the files a command writes under its --output folder
- `artifacts`     model recipe and results summary files
- `report`        plain-text rendering
- `plots`         matplotlib figures
- `logging_setup` log handlers: standard error plus <output>/knn-eval.log
"""

__version__ = "0.1.0"
