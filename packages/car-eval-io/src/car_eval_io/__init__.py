"""Application layer for the AME 505 Group 12 car-evaluation project.

- `__main__`   argparse entry point (`car-eval` / `python -m car_eval_io`)
- `app`        the commands: run, explain, show-config
- `config`     the YAML configuration schema and loader
- `converters` the cattrs converter that builds typed objects from YAML/JSON
- `sources`    where the raw dataset comes from and how it is read
- `artifacts`  model recipe and results summary files
- `report`     plain-text rendering
- `plots`      matplotlib figures
"""

__version__ = "0.1.0"
