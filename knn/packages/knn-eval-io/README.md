# knn-eval-io

Application layer of the k-NN model: YAML settings, object construction with cattrs, the
command line and every file the app reads or writes. The modelling lives in `knn-eval-core`.

| Module | Responsibility |
| --- | --- |
| `__main__` | argparse entry point (`knn-eval` or `python -m knn_eval_io`); owns every path: `--config`, `--data` / `--download`, `--output` |
| `paths` | `RunPaths`: the files a command writes under its `--output` folder |
| `logging_setup` | Log handlers: timestamped lines on standard error plus `<output>/knn-eval.log` |
| `app` | The commands: `run`, `explain`, `show-config` |
| `config` | The YAML schema (`AppConfig`: profile, figures, logging, pipeline, cars; no paths) and loader |
| `converters` | The cattrs converter: builds core objects from YAML/JSON and turns results back into data |
| `sources` | `LocalFile` / `UciDownload` and the raw CSV reader |
| `artifacts` | `model.json` recipe and `results.json` summary |
| `report` | Plain-text report for the console and `report.txt` |
| `plots` | Matplotlib figures |

```bash
uv run knn-eval run --config configs/knn.yaml --data ../data/car.data --output outputs
```

Tests: `uv run pytest packages/knn-eval-io` from the `knn` folder.
