from __future__ import annotations

import argparse
import pickle
import numpy as np
import json
import logging
import math
import glob
import random
import os
import sys
import datetime
import time

from functools import partial
from pathlib import Path
from collections import defaultdict
from shutil import copyfile
from tqdm import tqdm

import sklearn.model_selection
import sklearn.metrics

import torch

from ..extension.clmbr import *

from .. import timeline
from .. import ontology
from .. import labeler

from .dataset import DataLoader, convert_patient_data
from .prediction_model import CLMBR
from .trainer import Trainer
from .utils import read_config, read_info, device_from_config
from ..featurizer import ColumnValue, Featurizer
from ..splits import read_time_split
from ..utils import OnlineStatistics, set_up_logging

from .opt import OpenAIAdam

from typing import Mapping, Any, Dict, Optional, Tuple


def check_dir_for_overwrite(dirname: str) -> bool:
    return bool(
        glob.glob(os.path.join(dirname, "*.json"))
        or glob.glob(os.path.join(dirname, "checkpoints"))
    )


def create_info_program() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute training data summary statistics etc for CLMBR experiments"
    )
    parser.add_argument(
        "input_data_dir",
        type=str,
        help="Location of the dataset extract to be used for CLMBR training",
    )
    parser.add_argument(
        "save_dir", type=str, help="Location where model info is to be saved",
    )
    parser.add_argument(
        "train_end_date", type=str, help="The end date for training"
    )
    parser.add_argument(
        "val_end_date",
        type=str,
        help="The end date for validation. Should be later than the end date for training",
    )
    parser.add_argument(
        "--min_patient_count",
        type=int,
        default=100,
        help="Only keep statistics on codes/terms that appear for this many patients (default 100)",
    )
    parser.add_argument(
        "--excluded_patient_file",
        type=str,
        help="A file containing a list of patients to exclude from training. "
        "Any patient ID you plan to use for finetuning / evaluation should be "
        "listed in this file. If not provided, exclude_patient_ratio must be specified.",
        default=None,
    )
    parser.add_argument(
        "--train_patient_file",
        type=str,
        help="If provided, this will contain all of the patients allowed to be used for training",
        default=None,
    )
    parser.add_argument(
        "--val_patient_file",
        type=str,
        help="If provided, this will contain all of the patients allowed to be used for validation",
        default=None,
    )
    parser.add_argument(
        "--exclude_patient_ratio",
        type=float,
        default=None,
        help="Ratio of patients to exclude from pre-training between 0 and 1."
        " If provided, excluded patient IDs will "
        "be randomly selected and written out to a file "
        '"excluded_patient_ids.txt" in the save directory. If not '
        "provided, excluded_patient_file must be specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3451235,
        help="Random seed (default 3451235)",
    )
    parser.add_argument(
        "--train_start_date", type=str, help="The start date for training", default='1900-01-01'
    )
    parser.add_argument(
        "--val_start_date", type=str, help="The start date for validation", default='train_end_date'
    )
    args = parser.parse_args()

    if args.save_dir is None:
        print("Error - must specify save_dir", file=sys.stderr)
        exit(1)
    else:
        save_dir = args.save_dir

    if args.val_start_date == 'train_end_date':
        args.val_start_date = args.train_end_date

    os.makedirs(save_dir, exist_ok=True)

    set_up_logging(os.path.join(save_dir, "create_info.log"))
    logging.info("Args: %s", str(args))

    if check_dir_for_overwrite(save_dir):
        print(
            "Fatal error - model dir {} is not empty".format(save_dir),
            file=sys.stderr,
        )
        logging.info("Fatal error - model dir {} is not empty".format(save_dir))
        exit(1)

    ontologies_path = os.path.join(args.input_data_dir, "ontology.db")
    timelines_path = os.path.join(args.input_data_dir, "extract.db")

    train_start_date = datetime.datetime.fromisoformat(args.train_start_date)
    train_end_date = datetime.datetime.fromisoformat(args.train_end_date)
    
    if train_start_date >= train_end_date:
        logging.info("Training start date is after training end date?")
        exit(1)
    
    val_start_date = datetime.datetime.fromisoformat(args.val_start_date)
    val_end_date = datetime.datetime.fromisoformat(args.val_end_date)
    
    if val_start_date >= val_end_date:
        logging.info("Validation start date is after training end date?")
        exit(1)

    result = json.loads(
        create_info(
            timelines_path,
            ontologies_path,
            train_start_date,
            train_end_date,
            val_start_date,
            val_end_date,
            args.min_patient_count,
        )
    )
    result["extract_dir"] = args.input_data_dir
    result["extract_file"] = "extract.db"

    result["train_start_date"] = args.train_start_date
    result["train_end_date"] = args.train_end_date

    result["val_start_date"] = args.val_start_date
    result["val_end_date"] = args.val_end_date

    result["seed"] = args.seed
    result["min_patient_count"] = args.min_patient_count
    
    print('Starting point', len(result['val_patient_ids_with_length']))

    def remove_pids(a, x):
        return [(p, c) for p, c in a if p not in x]
    
    def require_pids(a, x):
        return [(p, c) for p, c in a if p in x]

    if args.excluded_patient_file is not None:
        with open(args.excluded_patient_file) as f:
            pids = {int(a) for a in f}

            result["train_patient_ids_with_length"] = remove_pids(
                result["train_patient_ids_with_length"], pids
            )
            result["val_patient_ids_with_length"] = remove_pids(
                result["val_patient_ids_with_length"], pids
            )
        logging.info(
            "Removed %d patient IDs from file %s"
            % (len(pids), args.excluded_patient_file)
        )
    elif args.exclude_patient_ratio is not None:
        assert 0 < args.exclude_patient_ratio and args.exclude_patient_ratio < 1
        train_pids = set([x[0] for x in result["train_patient_ids_with_length"]])
        val_pids = set([x[0] for x in result["val_patient_ids_with_length"]])
        all_pids = train_pids.union(val_pids)
        excluded_pids = set(
            random.sample(
                list(all_pids),
                int(round(len(all_pids) * args.exclude_patient_ratio)),
            )
        )

        result["train_patient_ids_with_length"] = remove_pids(
            result["train_patient_ids_with_length"], excluded_pids
        )
        result["val_patient_ids_with_length"] = remove_pids(
            result["val_patient_ids_with_length"], excluded_pids
        )
        with open(
            os.path.join(args.save_dir, "excluded_patient_ids.txt"), "w"
        ) as f:
            for pid in excluded_pids:
                f.write("%d\n" % pid)
        logging.info(
            "Removed %d patient IDs using ratio %f"
            % (len(excluded_pids), args.exclude_patient_ratio)
        )
    
    print('After exclusion', len(result['val_patient_ids_with_length']))

    if args.train_patient_file is not None:
        with open(args.train_patient_file) as f:
            pids = {int(a) for a in f}

            result["train_patient_ids_with_length"] = require_pids(
                result["train_patient_ids_with_length"], pids
            )
    
    if args.val_patient_file is not None:
        with open(args.val_patient_file) as f:
            pids = {int(a) for a in f}

            result["val_patient_ids_with_length"] = require_pids(
                result["val_patient_ids_with_length"], pids
            )
    
    print('Final', len(result['val_patient_ids_with_length']))

    def count_frequent_items(counts: Mapping[Any, int], threshold: int) -> int:
        return len(
            {item for item, count in counts.items() if count >= threshold}
        )

    logging.info(
        "Codes with >= 10 {}".format(
            count_frequent_items(result["code_counts"], 10)
        )
    )
    logging.info(
        "Codes with >= 25 {}".format(
            count_frequent_items(result["code_counts"], 25)
        )
    )
    logging.info(
        "Codes with >= 50 {}".format(
            count_frequent_items(result["code_counts"], 50)
        )
    )
    logging.info(
        "Codes with >= 100 {}".format(
            count_frequent_items(result["code_counts"], 100)
        )
    )
    logging.info(
        "Codes with >= 1000 {}".format(
            count_frequent_items(result["code_counts"], 1000)
        )
    )
    logging.info("Number codes: {}".format(len(result["code_counts"])))
    logging.info("Number valid codes: {}".format(len(result["valid_code_map"])))

    with open(os.path.join(args.save_dir, "info.json"), "w") as fp:
        json.dump(result, fp)


def train_model() -> None:
    parser = argparse.ArgumentParser(
        description="Representation Learning Experiments"
    )
    # paths
    parser.add_argument(
        "model_dir",
        type=str,
        help="Location where model logs and weights should be saved",
    )
    parser.add_argument(
        "info_dir",
        type=str,
        help="Location where `clmbr_create_info` results were saved",
    )
    parser.add_argument(
        "--extract_dir",
        action="store_true",
        help="Use the doctorai task definition",
    )

    # model specification
    parser.add_argument(
        "--size",
        default=768,
        type=int,
        help="Dimensionality of the output embeddings",
    )
    parser.add_argument(
        "--encoder_type",
        default="gru",
        choices=["gru", "lstm", "transformer"],
        help='the sequence encoder module type (default "gru")',
    )
    parser.add_argument("--no_tied_weights", default=False, action="store_true")
    parser.add_argument(
        "--rnn_layers",
        default=1,
        type=int,
        help='number of recurrent layers to use if encoder_type is "gru" or '
        '"lstm" (default 1), not used if encoder_type is "transformer"',
    )
    parser.add_argument(
        "--dropout",
        default=0,
        type=float,
        help="dropout percentage (default 0)",
    )

    # optimization specification
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Batch size (default 500)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2000,
        help="Batch size during evaluation (default 2000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default 50)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Number of warmup epochs (default 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default 0.01)"
    )
    parser.add_argument(
        "--l2",
        default=0.01,
        type=float,
        help="l2 regularization strength (default 0.01)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Specify whether the model should be run on CPU or GPU. Can specify a specific GPU, e.g. "cuda:0" (default "cpu")',
    )
    parser.add_argument("--code_dropout", type=float, default=0.2)
    # Day dropout added in reference to Lawrence's comment,
    # although Ethan mentioned it should be removed from the API
    parser.add_argument("--day_dropout", type=float, default=0.2)
    args = parser.parse_args()

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    if check_dir_for_overwrite(model_dir):
        print(
            "Fatal error - model dir {} is not empty".format(model_dir),
            file=sys.stderr,
        )
        logging.info(
            "Fatal error - model dir {} is not empty".format(model_dir)
        )
        exit(1)

    # Try to load info.json file; see create_info above for details.
    info = read_info(os.path.join(args.info_dir, "info.json"))
    copyfile(
        os.path.join(args.info_dir, "info.json"),
        os.path.join(model_dir, "info.json"),
    )

    first_too_small_index = float("inf")
    for code, index in info["valid_code_map"].items():
        if info["code_counts"][code] < 10 * info["min_patient_count"]:
            first_too_small_index = min(first_too_small_index, index)

    print(len(info["valid_code_map"]), flush=True)

    # Create and save config dictionary
    config = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_first": first_too_small_index,
        "num_second": len(info["valid_code_map"]) - first_too_small_index,
        "size": args.size,
        "lr": args.lr,
        "dropout": args.dropout,
        "encoder_type": args.encoder_type,
        "rnn_layers": args.rnn_layers,
        "tied_weights": not args.no_tied_weights,
        "l2": args.l2,
        "b1": 0.9,
        "b2": 0.999,
        "e": 1e-8,
        "epochs_per_cycle": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "code_dropout": args.code_dropout,
        "day_dropout": args.day_dropout,
        "model_dir": os.path.abspath(model_dir),
    }

    with open(os.path.join(model_dir, "config.json"), "w") as outfile:
        json.dump(config, outfile)

    set_up_logging(os.path.join(model_dir, "train.log"))
    logging.info("Args: %s", str(args))

    dataset = PatientTimelineDataset(
        os.path.join(info["extract_dir"], "extract.db"),
        os.path.join(info["extract_dir"], "ontology.db"),
        os.path.join(args.info_dir, "info.json"),
    )

    random.seed(info["seed"])

    model = CLMBR(config, info).to(torch.device(args.device))
    trainer = Trainer(model)
    trainer.train(dataset, use_pbar=False)


def debug_model() -> None:
    parser = argparse.ArgumentParser(
        description="Representation Learning Experiments"
    )
    parser.add_argument(
        "--model_dir", type=str, help="Override where model is saved"
    )
    args = parser.parse_args()

    model_dir = args.model_dir

    config = read_config(os.path.join(model_dir, "config.json"))
    info = read_info(os.path.join(model_dir, "info.json"))
    use_cuda = torch.cuda.is_available()

    model = CLMBR(config, info).to(device_from_config(use_cuda=use_cuda))
    model_data = torch.load(os.path.join(model_dir, "best"), map_location="cpu")
    model.load_state_dict(model_data)

    loaded_data = PatientTimelineDataset(
        os.path.join(info["extract_dir"], "extract.db"),
        os.path.join(info["extract_dir"], "ontology.db"),
        os.path.join(model_dir, "info.json"),
    )

    ontologies = ontology.OntologyReader(
        os.path.join(info["extract_dir"], "ontology.db")
    )
    timelines = timeline.TimelineReader(
        os.path.join(info["extract_dir"], "extract.db")
    )

    reverse_map = {}
    for b, a in info["valid_code_map"].items():
        word = ontologies.get_dictionary().get_word(b)
        reverse_map[a] = word

    reverse_map[len(info["valid_code_map"])] = "None"

    with DataLoader(
        loaded_data,
        threshold=config["num_first"],
        is_val=True,
        batch_size=1,
        seed=info["seed"],
        day_dropout=0,
        code_dropout=0,
    ) as batches:
        for batch in batches:
            if batch["task"][0].size()[0] == 0:
                continue
            values, non_text_loss = model(batch)
            values = torch.sigmoid(values)

            patient_id = int(batch["pid"][0])
            patient = timelines.get_patient(patient_id)
            original_day_indices = batch["day_index"][0]

            indices, targets, seen_before, _, _, _ = batch["task"]
            day_indices = indices[:, 0]
            word_indices = indices[:, 1]

            (
                all_non_text_codes,
                all_non_text_offsets,
                all_non_text_codes1,
                all_non_text_offsets1,
                all_day_information,
                all_positional_encoding,
                all_lengths,
            ) = batch["rnn"]

            all_non_text_codes = list(all_non_text_codes)
            all_non_text_offsets = list(all_non_text_offsets) + [
                len(all_non_text_codes)
            ]

            print(patient_id, batch["pid"], original_day_indices)

            all_seen = set()

            for i, index in enumerate(original_day_indices):
                day = patient.days[index]
                print("------------------")
                print(patient_id, i, index, day.age / 365, day.date)

                words = set()
                for code in day.observations:
                    for subword in ontologies.get_subwords(code):
                        words.add(ontologies.get_dictionary().get_word(subword))
                        all_seen.add(
                            ontologies.get_dictionary().get_word(subword)
                        )

                print("Source", words)

                wordsA = set()

                if (i + 1) < len(all_non_text_offsets):
                    for code in all_non_text_codes[
                        all_non_text_offsets[i] : all_non_text_offsets[i + 1]
                    ]:
                        wordsA.add(reverse_map[code.item()])

                print("Given", wordsA)

                day_mask = day_indices == i

                w = word_indices[day_mask]
                p = values[day_mask]
                t = targets[day_mask]
                f = seen_before[day_mask]

                items = [
                    (
                        t_i.item(),
                        reverse_map[w_i.item()],
                        p_i.item(),
                        reverse_map[w_i.item()] in all_seen,
                        w_i.item(),
                        f_i.item(),
                    )
                    for p_i, t_i, w_i, f_i in zip(p, t, w, f)
                ]
                items.sort(key=lambda x: (-x[0], x[1]))

                for a in items:
                    print(a)
