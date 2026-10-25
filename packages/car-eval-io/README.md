# car-eval-io

Application layer for the AME 505 Group 12 car-evaluation project: parsing, object
construction, the command line and every file the project reads or writes. The modelling
itself lives in `car-eval-core`.

| Module | Responsibility |
| --- | --- |
| `__main__` | argparse entry point (`car-eval` script, or `python -m car_eval_io`); owns every path: `--config`, `--data` / `--download`, `--output` |
| `paths` | `RunPaths`: the files a command writes under its `--output` folder |
| `app` | The commands: `run`, `explain`, `show-config` |
| `config` | The YAML configuration schema (`AppConfig`: profile, figures, pipeline, cars; no paths) and loader with profile merging |
| `converters` | The cattrs converter: builds core objects from YAML/JSON and turns results back into data |
| `sources` | `LocalFile` / `UciDownload` tagged union and the raw CSV reader |
| `artifacts` | `model.json` recipe (rebuilds the trained tree deterministically) and `results.json` summary |
| `report` | Plain-text rendering for the console and `report.txt` |
| `plots` | Matplotlib figures, each owned by a context manager that saves, shows and frees it |

```bash
uv run car-eval run --config configs/config.yaml --data data/car.data --output outputs
uv run car-eval explain --config configs/config.yaml --output outputs
uv run car-eval show-config --config configs/config.yaml
```

Exit codes: `0` success, `1` runtime failure (no saved model, data that does not match it, …),
`2` invalid configuration or command-line arguments (including a `--data` file that does not exist).

Tests: `uv run pytest packages/car-eval-io` from the repository root.
