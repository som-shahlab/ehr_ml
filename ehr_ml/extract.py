from __future__ import annotations

import argparse
import bisect
import os

from .extension.extract import (
    extract_omop
)


def extract_omop_program() -> None:
    parser = argparse.ArgumentParser(
        description="An extraction tool for OMOP v5 sources"
    )

    parser.add_argument(
        "omop_source",
        type=str,
        help="Path of the folder to the ehr_ml extraction",
    )

    parser.add_argument(
        "umls_location",
        type=str,
        help="The patient id to inspect",
    )
    
    parser.add_argument(
        "gem_location",
        type=str,
        help="The patient id to inspect",
    )

    parser.add_argument(
        "target_location",
        type=str,
        help="The patient id to inspect",
    )

    args = parser.parse_args()

    extract_omop(args.omop_source, args.umls_location, args.gem_location, args.target_location)