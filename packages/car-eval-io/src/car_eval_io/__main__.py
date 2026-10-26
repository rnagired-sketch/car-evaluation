"""Command-line entry point.

    uv run car-eval run --config configs/config.yaml --data data/car.data --output outputs
    uv run car-eval run --config configs/config.yaml --download --output outputs
    uv run car-eval explain --config configs/config.yaml --output outputs
    uv run car-eval show-config --config configs/config.yaml

`python -m car_eval_io ...` is equivalent. The YAML file says what to do (profile, pipeline
settings, figures, logging, cars). The command line says which file that is and every filesystem
location: where the data comes from and where results are written.

Progress and errors are log messages on standard error (and in `<output>/car-eval.log`); the
report itself goes to standard output. Argument mistakes are reported by argparse.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from car_eval_io import app
from car_eval_io.artifacts import ArtifactError
from car_eval_io.config import AppConfig, ConfigError, load_config
from car_eval_io.logging_setup import LogLevel, configure_logging
from car_eval_io.paths import RunPaths
from car_eval_io.sources import DataSource, LocalFile, UciDownload

COMMANDS: dict[str, str] = {
    "run": "run the full evaluation plan and write report, figures, results.json and model.json",
    "explain": "rate and explain the config's `cars` with the model saved by a previous run",
    "show-config": "print the resolved configuration, defaults included",
}
DEFAULT_OUTPUT = Path("outputs")

EXIT_OK, EXIT_FAILURE, EXIT_BAD_INPUT, EXIT_INTERRUPTED = 0, 1, 2, 130

logger = logging.getLogger("car_eval_io")


def existing_file(text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"no such file: {text}")
    return path.resolve()


def folder(text: str) -> Path:
    return Path(text).expanduser().resolve()


def _add_config(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "-c", "--config", type=Path, required=True, metavar="FILE", help="YAML configuration file"
    )


def _add_output(command: argparse.ArgumentParser, help_text: str) -> None:
    command.add_argument(
        "-o",
        "--output",
        type=folder,
        default=str(DEFAULT_OUTPUT),  # a string default goes through `type`, so it is resolved too
        metavar="DIR",
        help=f"{help_text} (default: %(default)s)",
    )


def _add_data(command: argparse.ArgumentParser, required: bool, default_note: str = "") -> None:
    source = command.add_mutually_exclusive_group(required=required)
    source.add_argument(
        "-d", "--data", type=existing_file, metavar="FILE", help=f"local car.data file{default_note}"
    )
    source.add_argument("--download", action="store_true", help="download car.data from the UCI repository")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="car-eval",
        description="AME 505 Group 12: decision-tree car evaluation. Settings come from a YAML "
        "file; data and output locations come from these options.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser("run", help=COMMANDS["run"], description=COMMANDS["run"])
    _add_config(run)
    _add_data(run, required=True)
    _add_output(run, "folder for report.txt, results.json, model.json and figures/")

    explain = commands.add_parser("explain", help=COMMANDS["explain"], description=COMMANDS["explain"])
    _add_config(explain)
    _add_data(explain, required=False, default_note=" (default: the source recorded in model.json)")
    _add_output(explain, "folder holding model.json from a previous run; explanations.json is written here")

    show = commands.add_parser(
        "show-config", help=COMMANDS["show-config"], description=COMMANDS["show-config"]
    )
    _add_config(show)
    return parser


def data_source(args: argparse.Namespace) -> DataSource | None:
    """The source named on the command line, or None when neither option was given."""
    if getattr(args, "download", False):
        return UciDownload()
    path: Path | None = getattr(args, "data", None)
    return LocalFile(path) if path is not None else None


def output_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RunPaths:
    try:
        return RunPaths(args.output)
    except ValueError as error:
        parser.error(str(error))  # exits with status 2, like every other argument error


def log_file(config: AppConfig, paths: RunPaths | None) -> Path | None:
    return paths.log if paths is not None and config.logging.save else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(LogLevel.INFO)  # until the config says otherwise
    try:
        config = load_config(args.config)
    except ConfigError as error:
        logger.error("%s", error)
        return EXIT_BAD_INPUT

    paths = output_paths(parser, args) if args.command in ("run", "explain") else None
    configure_logging(config.logging.level, log_file(config, paths))
    logger.debug("command %r with config %s", args.command, args.config)

    try:
        match args.command:
            case "run":
                data = data_source(args)
                if data is None or paths is None:
                    raise AssertionError("argparse requires --data or --download and --output for `run`")
                app.run(config, data, paths)
            case "explain":
                if paths is None:
                    raise AssertionError("`explain` always has an output folder")
                app.explain(config, paths, data_source(args))
            case "show-config":
                app.show_config(config)
            case other:
                raise AssertionError(f"unhandled command {other!r}")
    except (ArtifactError, OSError, ValueError) as error:
        logger.error("%s", error)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        logger.warning("interrupted; results written so far are kept")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("unexpected error; the traceback below shows where it happened")
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
