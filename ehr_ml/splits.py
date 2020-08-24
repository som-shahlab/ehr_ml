from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from typing import Any, List, Tuple

from . import timeline


def read_patient_split(
    extract_path: str, fold: str, split_type: str = "patient_splits"
) -> Tuple[List[int], List[int], int]:
    """Read the information associated with a patient split.

    Args:
        extract_path (str): The path to the stride_ml extract.
        fold (int): An integer specifying the split number.

    Returns:
        A tuple containing a list of the training patient_ids, the testing patient_ids and the integer seed.
    """
    train_ids = read_id_file(
        os.path.join(extract_path, split_type, str(fold), "train_ids")
    )
    test_ids = read_id_file(
        os.path.join(extract_path, split_type, str(fold), "test_ids")
    )
    seed = read_seed(os.path.join(extract_path, split_type, str(fold), "seed"))

    return train_ids, test_ids, seed


def read_time_split(
    extract_path: str, split_name: str
) -> Tuple[datetime.date, datetime.date, int]:
    """Read the information associated with a patient split.

    Args:
        extract_path (str): The path to the stride_ml extract.
        split_name (str): A string containing the name of the split.

    Returns:
        A tuple containing the start date of the split, the end date of the split and the corresponding integer seed.
    """
    with open(
        os.path.join(extract_path, "time_splits", split_name, "START")
    ) as input_fd:
        year, month, day = input_fd.readline().rstrip().split("-")
    start_date = datetime.date(year=int(year), month=int(month), day=int(day))

    with open(
        os.path.join(extract_path, "time_splits", split_name, "END")
    ) as input_fd:
        year, month, day = input_fd.readline().rstrip().split("-")
    end_date = datetime.date(year=int(year), month=int(month), day=int(day))

    seed = read_seed(
        os.path.join(extract_path, "time_splits", split_name, "seed")
    )

    return start_date, end_date, seed


def read_split_directory(extract_path: str, split_dir: str) -> Tuple[str, Any]:
    """Read the information associated with a generic split directory"""

    split_type, split_name = split_dir.split("/")

    if split_type in ("patient_splits", "cross_validation"):
        return (
            "patient",
            read_patient_split(extract_path, split_name, split_type=split_type),
        )
    elif split_type == "time_splits":
        return "time", read_time_split(extract_path, split_name)
    else:
        raise ValueError(f"Invalid split type {split_type}")


def read_id_file(filename: str) -> List[int]:
    ids = []
    with open(filename) as file:
        for line in file:
            ids.append(int(line))
    return ids


def read_seed(filename: str) -> int:
    with open(filename) as file:
        return int(file.read())


def create_splits() -> None:
    parser = argparse.ArgumentParser(description="Create splits for ehr_ml")
    parser.add_argument(
        "--extract_path",
        type=str,
        help="",
        default=os.environ.get("EHR_ML_EXTRACT_DIR"),
    )
    args = parser.parse_args()
    print(
        "Writing splits to {}/patient_splits".format(args.extract_path),
        file=sys.stderr,
    )
    true_random = random.SystemRandom()

    timelines = timeline.TimelineReader(
        os.path.join(args.extract_path, "extract.db")
    )
    patient_ids = list(timelines.get_patient_ids())
    patient_ids.sort()

    for fold in range(10):
        patient_ids_copy = list(patient_ids)

        random_seed = true_random.getrandbits(63)

        random.seed(random_seed)
        random.shuffle(patient_ids_copy)

        num_train = int(0.7 * len(patient_ids_copy))

        train_set = set(patient_ids_copy[:num_train])

        train_ids = [
            pid for pid in timelines.get_patient_ids() if pid in train_set
        ]
        test_ids = [
            pid for pid in timelines.get_patient_ids() if pid not in train_set
        ]

        directory = os.path.join(args.extract_path, "patient_splits", str(fold))

        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "seed"), "w") as file:
            file.write(str(random_seed) + "\n")

        with open(os.path.join(directory, "train_ids"), "w") as file:
            for t in train_ids:
                file.write(str(t) + "\n")

        with open(os.path.join(directory, "test_ids"), "w") as file:
            for t in test_ids:
                file.write(str(t) + "\n")
