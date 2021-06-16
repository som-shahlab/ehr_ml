from __future__ import annotations

import argparse
import bisect
import os

from .extension.extract import extract_omop


def extract_omop_program() -> None:
    parser = argparse.ArgumentParser(
        description="An extraction tool for OMOP v5 sources"
    )

    parser.add_argument(
        "omop_source", type=str, help="Path of the folder to the omop source",
    )

    parser.add_argument(
        "umls_location", type=str, help="The location of the umls directory",
    )

    parser.add_argument(
        "gem_location", type=str, help="The location of the gem directory",
    )

    parser.add_argument(
        "rxnorm_location",
        type=str,
        help="The location of the rxnorm directory",
    )

    parser.add_argument(
        "target_location", type=str, help="The place to store the extract",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="The delimiter used in the raw OMOP source",
    )

    parser.add_argument(
        "--ignore_quotes",
        dest="use_quotes",
        action="store_false",
        help="Ignore quotes while parsing",
    )
    parser.add_argument(
        "--use_quotes",
        dest="use_quotes",
        action="store_true",
        help="Use quotes while parsing",
    )
    parser.set_defaults(use_quotes=True)

    args = parser.parse_args()

    print(args)

    extract_omop(
        args.omop_source,
        args.umls_location,
        args.gem_location,
        args.rxnorm_location,
        args.target_location,
        args.delimiter,
        args.use_quotes,
    )
