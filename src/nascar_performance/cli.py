"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nascar_performance.config import sportsdataio_key
from nascar_performance.data import SportsDataClient, raw_to_tables
from nascar_performance.demo import generate_demo_data
from nascar_performance.features import build_historical_features, load_modeling_data
from nascar_performance.modeling import save_artifact, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nascar-performance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Download a completed season")
    collect.add_argument("--season", type=int, required=True)
    collect.add_argument("--output-dir", type=Path, default=Path("data/raw"))

    prepare = subparsers.add_parser("prepare", help="Convert raw JSON to CSV tables")
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    demo = subparsers.add_parser("demo-data", help="Generate deterministic synthetic data")
    demo.add_argument("--output-dir", type=Path, default=Path("data/demo"))

    train = subparsers.add_parser("train", help="Train and evaluate the ranking model")
    train.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    train.add_argument("--model-out", type=Path, default=Path("models/nascar_rank_model.joblib"))
    train.add_argument("--metrics-out", type=Path, default=Path("models/metrics.json"))
    train.add_argument(
        "--normalize-scrambled",
        action="store_true",
        help="Replace order-preserving provider values with within-race ranks",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "collect":
        result = SportsDataClient(sportsdataio_key()).collect_season(args.season, args.output_dir)
    elif args.command == "prepare":
        result = raw_to_tables(args.raw_dir, args.output_dir)
    elif args.command == "demo-data":
        result = generate_demo_data(args.output_dir)
    else:
        frame = load_modeling_data(args.data_dir, args.normalize_scrambled)
        featured = build_historical_features(frame)
        artifact, metadata = train_model(featured)
        save_artifact(artifact, metadata, args.model_out, args.metrics_out)
        result = metadata
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
