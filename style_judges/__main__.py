"""Command-line entry points for the experiment."""

import argparse
from pathlib import Path

from style_judges.protocol import EXTENSION_STAGES, RESULTS
from style_judges.sources import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    run = commands.add_parser("run")
    analyze = commands.add_parser("analyze")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    for command in (prepare, run, analyze, worker):
        command.add_argument("--output", type=Path, default=RESULTS)
    for command in (run, worker):
        splits = ("development", "heldout")
        if command is worker:
            splits += EXTENSION_STAGES
        command.add_argument("split", choices=splits)
        command.add_argument("precision", choices=("nf4", "bf16"), default="nf4", nargs="?")
    worker.add_argument("model", choices=list(MODELS))
    run.add_argument("--model", choices=list(MODELS), help="Run one development checkpoint")
    analyze.add_argument("--precision", choices=("nf4", "bf16"), default="nf4")
    args = parser.parse_args()
    root = args.output.resolve()

    if args.command == "prepare":
        from style_judges.data import prepare

        prepare(root)
    elif args.command == "run":
        from style_judges.inference import supervise

        supervise(root, args.split, args.precision, model=args.model)
    elif args.command == "_worker":
        from style_judges.inference import worker

        worker(root, args.split, args.precision, args.model)
    else:
        from style_judges.analysis import analyze

        analyze(root, args.precision)


if __name__ == "__main__":
    main()
