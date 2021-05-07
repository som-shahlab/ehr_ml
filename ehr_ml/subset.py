from __future__ import annotations

import argparse
import bisect
import os

from .extension.subset import extract_subset


def extract_subset_program() -> None:
    parser = argparse.ArgumentParser(
        description="A tool to generate a subset of an existing ehr_ml extract"
    )

    parser.add_argument(
        "src_timeline_path", type=str, help="Path of the source extract"
    )

    parser.add_argument(
        "target_timeline_path", type=str, help="Path of the destination extract"
    )

    parser.add_argument(
        "subset_ratio", type=float, help="subset ratio of the existing extract"
    )

    parser.add_argument(
        "--seed", type=int, help="random seed for sampling", default=42
    )

    args = parser.parse_args()

    print(args)

    extract_subset(
        args.src_timeline_path,
        args.target_timeline_path,
        args.subset_ratio,
        args.seed,
    )
