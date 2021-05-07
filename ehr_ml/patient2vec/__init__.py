import argparse
import datetime
import glob
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

from .. import labeler, ontology, timeline
from ..extension.patient2vec import *
from ..featurizer import ColumnValue, Featurizer
from ..splits import read_time_split
from ..utils import OnlineStatistics, set_up_logging
from . import dataset
from .lamb import Lamb
from .opt import OpenAIAdam
from .prediction_model import PredictionModel


def check_dir_for_overwrite(dirname: str) -> bool:
    return bool(
        glob.glob(os.path.join(dirname, "*.json"))
        or glob.glob(os.path.join(dirname, "checkpoints"))
    )


def create_info_program() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute training data summary statistics etc for patient2vec experiments"
    )
    parser.add_argument(
        "--extract_dir",
        default=os.environ.get("EHR_ML_EXTRACT_DIR"),
        type=str,
        help="Extract dir; overrides environment var $EHR_ML_EXTRACT_DIR",
    )
    parser.add_argument(
        "--extract_file",
        type=str,
        default="extract.db",
        help="Name of extract file in --extract_dir",
    )
    parser.add_argument(
        "--min_patient_count",
        type=int,
        default=100,
        help="Only keep statistics on codes/terms that appear for this many patients",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override dir where model is saved",
    )
    args = parser.parse_args()

    if args.save_dir is None:
        print("Error - must specify save_dir", file=sys.stderr)
        exit(1)
    else:
        save_dir = args.save_dir

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

    ontologies_path = os.path.join(args.extract_dir, "ontology.db")
    timelines_path = os.path.join(args.extract_dir, args.extract_file)

    seed = 45234123

    result = json.loads(
        create_info(
            timelines_path, ontologies_path, seed, args.min_patient_count
        )
    )
    result["extract_dir"] = args.extract_dir
    result["extract_file"] = args.extract_file
    result["seed"] = seed
    result["min_patient_count"] = args.min_patient_count

    first_too_small_index = float("inf")
    for code, index in result["valid_code_map"].items():
        if result["code_counts"][code] < 10 * result["min_patient_count"]:
            first_too_small_index = min(first_too_small_index, index)

    result["threshold"] = first_too_small_index

    print(
        f'The first bin consists of {first_too_small_index} codes with {len(result["valid_code_map"]) - first_too_small_index} remaining'
    )

    def count_frequent_items(counts: Mapping[Any, int], threshold: int) -> int:
        return len(
            {item for item, count in counts.items() if count >= threshold}
        )

    for t in (10, 25, 50, 100, 250, 500, 1000, 5000):
        logging.info(
            "Codes with >= {} {}".format(
                t, count_frequent_items(result["code_counts"], t)
            )
        )
    logging.info("Number codes: {}".format(len(result["code_counts"])))
    logging.info("Number valid codes: {}".format(len(result["valid_code_map"])))

    with open(os.path.join(args.save_dir, "info.json"), "w") as fp:
        json.dump(result, fp)


def read_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as file:
        config = json.load(file)
        return config


def read_info(info_file: str) -> Dict[str, Any]:
    if not Path(os.path.join(info_file)).is_file():
        print("Fatal error: info.json not found.", file=sys.stderr)
        logging.info("Fatal error: info.json not found")
        exit(1)

    def date_from_str(x: str) -> Optional[datetime.date]:
        if x == "None":
            return None
        else:
            date_obj = datetime.datetime.strptime(x, "%Y-%m-%d").date()
            return date_obj

    with open(info_file, "rb") as file:
        info = json.load(file)
        info["valid_code_map"] = {
            int(code): int(idx) for code, idx in info["valid_code_map"].items()
        }
        info["code_counts"] = {
            int(code): int(idx) for code, idx in info["code_counts"].items()
        }
        for date_name in [
            "train_start_date",
            "train_end_date",
            "val_start_date",
            "val_end_date",
        ]:
            if date_name in info:
                info[date_name] = date_from_str(info[date_name])
        return info


def device_from_config(use_cuda: bool) -> torch.device:
    return torch.device("cuda:0" if use_cuda else "cpu")


def train_model_func(args, rank, size):
    if args.info_dir is None or not Path(args.info_dir).is_dir():
        print("Error - must provide path to info directory", file=sys.stderr)
        exit(1)

    if args.model_dir is None:
        print("Error - must provide model dir", file=sys.stderr)
        exit(1)

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    epochs_per_cycle = 1
    warmup_epochs = 0.01
    # epochs_per_cycle = 30
    # warmup_epochs = 1

    # Try to load info.json file; see create_info above for details.
    info = read_info(os.path.join(args.info_dir, "info.json"))
    copyfile(
        os.path.join(args.info_dir, "info.json"),
        os.path.join(model_dir, "info.json"),
    )

    set_up_logging(os.path.join(model_dir, "train.log"))
    logging.info("Args: %s", str(args))

    info["extract_dir"] = "/local-scratch/nigam/ehr_ml_extracts/zip_optum/0_1"

    print(info["extract_dir"])

    loaded_data = StrideDataset(
        os.path.join(info["extract_dir"], "extract.db"),
        os.path.join(info["extract_dir"], "ontology.db"),
        os.path.join(args.model_dir, "info.json"),
    )
    # Create and save config dictionary
    config = {
        "batch_size": args.batch_size,
        "num_first": info["threshold"] + info["num_lab_codes"],
        "num_second": len(info["valid_code_map"]) - info["threshold"],
        "num_valid_targets": 5001,
        "size": args.size,
        "lr": args.lr,
        "dropout": args.dropout,
        "use_gru": args.use_gru,
        "gru_layers": args.gru_layers,
        "gru_hidden_size": args.gru_hidden_size,
        "tied_weights": not args.no_tied_weights,
        "l2": args.l2,
        "b1": 0.9,
        "b2": 0.999,
        "e": 1e-8,
        "epochs_per_cycle": epochs_per_cycle,
        "warmup_epochs": warmup_epochs,
        "code_dropout": args.code_dropout,
    }

    with open(os.path.join(model_dir, "config.json"), "w") as outfile:
        json.dump(config, outfile)

    seed = (info["seed"] // (2 ** 32)) + 14220 * rank + 34342
    print("Got seed", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda

    model = PredictionModel(config, info, use_cuda=use_cuda).to(
        device_from_config(use_cuda=use_cuda)
    )

    if True:
        baseline = "large_common_factor_speed/"
        # baseline = 'survival_model_approx_a_3'
        print("Working off", baseline)
        model_data = torch.load(
            os.path.join(baseline, "best"), map_location="cpu"
        )
        model.load_state_dict(model_data)
    else:
        print("Not working")

    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # if name == 'module.timeline_model.model.encoder.layers.11.linear2.weight':
        #     print(name, param[0])
        params.append(param)

    final_transformation = partial(
        PredictionModel.finalize_data,
        config,
        info,
        device_from_config(use_cuda=use_cuda),
    )

    logging.info(
        "Iters per epoch %s",
        loaded_data.num_train_batches(config["batch_size"]),
    )

    # optimizer = torch.optim.SGD(params, lr=config['lr'], momentum=0.9)

    logging.info("Working with learning rate: %s", config["lr"])

    # print('Lamb')
    # optimizer = Lamb(params, lr=config['lr'])
    print("Adam")
    optimizer = torch.optim.Adam(params, lr=config["lr"])

    # OpenAIAdam(params, lr=config['lr'], schedule='warmup_linear', warmup=config['warmup_epochs']/config['epochs_per_cycle'],
    #        t_total=loaded_data.num_train_batches(config['batch_size']) * config['epochs_per_cycle'], b1=config['b1'], b2=config['b2'], e=config['e'], l2=config['l2'])

    def train_epoch() -> Iterable[None]:
        model.train()

        all_non_text_loss = 0

        last_time = time.time()

        code_dropout = 0.0
        day_dropout = 0.0
        print(f"Code dropout is {code_dropout}")
        print(f"Day dropout is {day_dropout}")

        with dataset.BatchIterator(
            loaded_data,
            final_transformation,
            is_val=False,
            batch_size=config["batch_size"],
            seed=random.randint(0, 100000),
            day_dropout=day_dropout,
            code_dropout=code_dropout,
        ) as batches:
            for i, batch in enumerate(batches):

                # for name, param in model.named_parameters():
                #     print(name, param.sum())

                values, non_text_loss = model(batch)
                del values

                non_text_loss.backward()

                # print(non_text_loss.item())
                # non_text_loss.backward()

                # print(model.task_module.scale_and_scale_function.weight.grad.shape)
                # print(model.task_module.scale_and_scale_function.weight.grad)

                # g = model.task_module.scale_and_scale_function.weight.grad.view(800, 5001, 2)
                # perm_g = g.permute((1, 0, 2))
                # print(perm_g.shape)

                # perm_g = perm_g.reshape((5001, 800 * 2))

                # normed = torch.norm(perm_g, dim=1)

                # print(normed.shape)

                # print(normed)

                # average_norm = torch.mean(normed)

                # for i, val in enumerate(normed):
                #     print(i, val / average_norm)

                # print(1/0)

                warmup = 5e3

                if i < warmup:
                    scale = (i / warmup) * (1 / math.sqrt(warmup))
                else:
                    scale = 1 / math.sqrt(i)

                for param_group in optimizer.param_groups:
                    param_group["lr"] = scale * config["lr"]

                # for name, param in model.named_parameters():
                #     print(name, param.sum(), param.grad.sum() if param.grad is not None else 'No grad')

                optimizer.step()
                optimizer.zero_grad()

                del non_text_loss
                del batch

                if i != 0 and i % 1000 == 0:
                    current_time = time.time()
                    delta = current_time - last_time
                    if i != 0:
                        print(
                            "Iters per second ",
                            1000 / delta,
                            " ",
                            i,
                            scale * config["lr"],
                        )
                    last_time = current_time

                if i != 0 and i % 3000 == 0:
                    yield
                    model.train()

        yield

    def test(sample_size: int = 100) -> Tuple[float, float]:
        model.eval()
        train_loss = test_helper(is_val=False, sample_size=sample_size)
        val_loss = test_helper(is_val=True, sample_size=sample_size)
        return train_loss, val_loss

    def test_helper(is_val: bool, sample_size: int) -> float:
        non_text_total_loss = 0

        with dataset.BatchIterator(
            loaded_data,
            final_transformation,
            is_val=is_val,
            batch_size=2000,
            seed=0,
            day_dropout=0,
            code_dropout=0,
        ) as batches:
            for batch, i in zip(batches, range(sample_size)):
                with torch.no_grad():
                    values, non_text_loss = model(batch)
                    del values

                    non_text_total_loss += non_text_loss.item()

                    del batch
                    del non_text_loss

        return non_text_total_loss / sample_size

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = None
    best_val_loss_epoch = None

    partial_epoch_index = 0

    logging.info("Start training")
    with open(os.path.join(model_dir, "losses"), "w") as loss_file:
        for epoch in range(config["epochs_per_cycle"]):
            logging.info("About to start epoch %s", epoch)
            for _ in train_epoch():
                partial_epoch_index += 1

                logging.info(
                    "Partial epoch is complete %s %s",
                    epoch,
                    partial_epoch_index,
                )
                for name, param in model.named_parameters():
                    if (
                        name
                        == "module.timeline_model.model.encoder.layers.11.linear2.weight"
                    ):
                        print(name, param[0])

                train_loss, val_loss = test(sample_size=2000)

                logging.info("Train loss: %s", train_loss)
                logging.info("Val loss: %s", val_loss)

                loss_file.write(
                    "Epoch {} {}\n".format(epoch, partial_epoch_index)
                )
                loss_file.write("Train loss {}\n".format(train_loss))
                loss_file.write("Val loss {}\n".format(val_loss))
                loss_file.write("\n")
                loss_file.flush()

                if rank == 0 and (
                    best_val_loss is None or val_loss < best_val_loss
                ):
                    best_val_loss = val_loss

                    if os.path.exists(os.path.join(model_dir, "best")):
                        os.unlink(os.path.join(model_dir, "best"))

                    print("saving")
                    torch.save(
                        model.state_dict(), os.path.join(model_dir, "best")
                    )


def init_process(rank, args, size, fn):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=size
    )
    fn(args, rank, size)
    torch.distributed.destroy_process_group()


def train_model() -> None:
    parser = argparse.ArgumentParser(
        description="Representation Learning Experiments"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Override where model is saved",
    )
    parser.add_argument("--info_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--size", default=768, type=int,
    )
    parser.add_argument(
        "--use_gru", default=False, action="store_true",
    )
    parser.add_argument("--no_tied_weights", default=False, action="store_true")
    parser.add_argument("--gru_layers", default=1, type=int)
    parser.add_argument("--gru_hidden_size", default=768, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--l2", default=0.01, type=float)
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Batch size"
    )
    parser.add_argument(
        "--extract_dir",
        action="store_true",
        help="Use the doctorai task definition",
    )
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--code_dropout", type=float, default=0.0)
    args = parser.parse_args()

    train_model_func(args, 0, 1)

    # size = 1
    # torch.multiprocessing.spawn(init_process, args=(args, size, train_model_func), nprocs=size)


def debug_model() -> None:
    parser = argparse.ArgumentParser(
        description="Representation Learning Experiments"
    )
    parser.add_argument(
        "--model_dir", type=str, help="Override where model is saved"
    )
    parser.add_argument("--use_train", default=False, action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir

    config = read_config(os.path.join(model_dir, "config.json"))
    info = read_info(os.path.join(model_dir, "info.json"))

    use_cuda = torch.cuda.is_available()

    model = PredictionModel(config, info, use_cuda=use_cuda).to(
        device_from_config(use_cuda=use_cuda)
    )
    model_data = torch.load(os.path.join(model_dir, "best"), map_location="cpu")
    model.load_state_dict(model_data)

    if os.path.exists("/local-scratch/nigam/ehr_ml_extracts/optum_zip/0_1"):
        info[
            "extract_dir"
        ] = "/local-scratch/nigam/ehr_ml_extracts/optum_zip/0_1"

    final_transformation = partial(
        PredictionModel.finalize_data,
        config,
        info,
        device_from_config(use_cuda=use_cuda),
    )

    ontologies = ontology.OntologyReader(
        os.path.join(info["extract_dir"], "ontology.db")
    )
    timelines = timeline.TimelineReader(
        os.path.join(info["extract_dir"], "extract.db")
    )

    reverse_map = {}
    reverse_map_backup = {}

    for b, a in info["valid_code_map"].items():
        word = ontologies.get_dictionary().get_word(b)
        if a < info["threshold"]:
            reverse_map[a] = word
        else:
            reverse_map_backup[a - info["threshold"]] = "BACKUP/" + word

    reverse_target_map = {}
    for b, a in info["valid_target_map"].items():
        word = ontologies.get_dictionary().get_word(int(b))
        reverse_target_map[a] = word
    reverse_target_map[5000] = "Censor"

    for a, b in info["lab_value_map"].items():
        code = int(a)
        word = timelines.get_dictionary().get_word(code)
        if "numeric_indices" in b:
            numeric_ranges = (
                [float("-inf")] + b["numeric_ranges"] + [float("inf")]
            )
            numeric_indices = b["numeric_indices"]

            for i, elem in enumerate(numeric_indices):
                text_value = (
                    f"{word}: ({numeric_ranges[i]}-{numeric_ranges[i+1]})"
                )
                reverse_map[info["threshold"] + elem] = text_value

        if "text_indices" in b:
            for k, elem in b["text_indices"].items():
                value = b["text_values"][k]
                text_value = f"{word}: {value}"
                reverse_map[info["threshold"] + elem] = text_value

    reverse_map[info["threshold"] + info["num_lab_codes"]] = "None"
    reverse_map_backup[len(reverse_map_backup)] = "BACKUP/None"

    model.eval()

    patient_labels = np.load("better_example3/treatments.npy")
    patient_ids = np.load("better_example3/patient_ids.npy")
    patient_indices = np.load("better_example3/patient_indices.npy")

    data = (patient_labels, patient_ids, patient_indices)

    print("starting to load")

    loaded_data = StrideDataset(
        os.path.join(info["extract_dir"], "extract.db"),
        os.path.join(info["extract_dir"], "ontology.db"),
        os.path.join(model_dir, "info.json"),
        # )
        data,
        data,
    )

    patient_id_to_info = {}

    for i, (pid, index) in enumerate(zip(patient_ids, patient_indices)):
        patient_id_to_info[pid] = (index, i)

    patient_representations = np.zeros((len(patient_ids), 800))

    print("working")
    # batch_size = 1
    batch_size = 10000

    with dataset.BatchIterator(
        loaded_data,
        final_transformation,
        is_val=not args.use_train,
        batch_size=batch_size,
        seed=info["seed"],
        day_dropout=0,
        code_dropout=0,
    ) as batches:
        for batch in batches:

            with torch.no_grad():
                embeddings = (
                    model.compute_embedding_batch(batch["rnn"]).cpu().numpy()
                )
                for i, patient_id in enumerate(batch["pid"]):
                    index, target_i = patient_id_to_info[patient_id]
                    patient_representations[target_i, :] = embeddings[
                        i, index, :
                    ]
                print("Got batch done", len(batch["pid"]))
            continue

            if batch["task"][0].size()[0] == 0:
                continue

            logits, non_text_loss = model(batch)
            probs = torch.sigmoid(logits)

            # print(values, loss)

            # values = values.cpu().detach().numpy()
            # loss = loss.cpu().detach().numpy()

            patient_id = int(batch["pid"][0])
            patient = timelines.get_patient(patient_id)

            original_day_indices: List[int] = batch["day_index"][0]

            target_indices, target_labels, target_factors, _ = batch["task"]
            print(
                target_indices.shape, target_labels.shape, target_factors.shape
            )

            day_to_targets_map = defaultdict(list)

            for i in range(len(target_labels)):
                if i < len(target_factors):
                    factor = target_factors[i].item()
                else:
                    factor = 1.0

                day_offset = target_indices[i, 0].item()
                target_id = target_indices[i, 1].item()
                target_label = target_labels[i].item()
                prob = probs[i].item() * factor

                def safe_log(a: float) -> float:
                    if a == 0:
                        return float("-inf")
                    else:
                        return math.log(a)

                if target_label == 1:
                    loss = safe_log(prob)
                else:
                    loss = safe_log(1 - prob)

                # print(day_offset, target_id, target_label, factor)

                day_to_targets_map[day_offset].append(
                    (target_id, target_label, factor, prob, loss)
                )

            # times = times.cpu().detach().numpy()
            # censor = censor.cpu().detach().numpy()
            # event = event.cpu().detach().numpy()

            # print(times)
            # print(censor)
            # print(event)

            # print(np.mean(censor))
            # print(np.mean(event))

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

            all_non_text_codes1 = list(all_non_text_codes1)
            all_non_text_offsets1 = list(all_non_text_offsets1) + [
                len(all_non_text_codes1)
            ]

            print("Backup codes", len(all_non_text_codes1))
            print("Got loss", non_text_loss.item())

            print(patient_id, batch["pid"], original_day_indices)

            for i, index in enumerate(original_day_indices):
                day = patient.days[index]
                print("------------------")
                print(patient_id, i, index, day.age / 365, day.date)

                words = set()
                for code in day.observations:
                    for subword in ontologies.get_subwords(code):
                        words.add(ontologies.get_dictionary().get_word(subword))

                for code_with_obs in day.observations_with_values:
                    name = timelines.get_dictionary().get_word(
                        code_with_obs.code
                    )
                    if code_with_obs.is_text:
                        value = timelines.get_value_dictionary().get_word(
                            code_with_obs.text_value
                        )
                        words.add(f"{name}: {value}")
                    else:
                        words.add(f"{name}: {code_with_obs.numeric_value}")

                print("Source", words)

                wordsA = set()

                if (i + 1) < len(all_non_text_offsets):
                    for code_value in all_non_text_codes[
                        all_non_text_offsets[i] : all_non_text_offsets[i + 1]
                    ]:
                        wordsA.add(reverse_map[code_value.item()])

                    for code_value in all_non_text_codes1[
                        all_non_text_offsets1[i] : all_non_text_offsets1[i + 1]
                    ]:
                        wordsA.add(reverse_map_backup[code_value.item()])

                print("Given", wordsA)

                def frac(a, b):
                    asum = np.sum(a)
                    bsum = np.sum(b)
                    return asum / (asum + bsum)

                # print('unweighted', frac(event[0, i, :], censor[0, i, :]))

                for target, label, factor, prob, loss in day_to_targets_map[i]:
                    target_name_part = target // (20 * 2)
                    target_sub_part = (target // 20) % 2
                    target_time_part = (target) % (20)
                    target_name = reverse_target_map[target_name_part]

                    if target_name != "ICD10CM/E00-E89":
                        continue

                    print(
                        target_name,
                        target_sub_part,
                        target_time_part,
                        label,
                        factor,
                        prob,
                        loss,
                    )

                    # if event[0, i, target] == 0 and censor[0, i, target] == 0:
                    #     continue

                    # if event[0, i, target] == 1 or target_name.startswith('ICD10CM/J0'): #or target_name == 'ICD10CM/E11':
                    #     # print(target_name, event[0, i, target], times[0, i, target], shapes[0, i, target], scale[0, i, target], medians[0, i, target], prob_censor[target], weight[target], weight2[target], loss[0, i, target])
                    #     print(target_name, event[0, i, target], times[0, i, target], values[0, i, target, :].tolist(), loss[0, i, target])

    np.save("better_example3/repr.npy", patient_representations)


def mass_featurizer(
    model_dir: str,
    labelers: Mapping[str, labeler.SavedLabeler],
    no_cuda: bool = False,
) -> Tuple[Dict[str, Dict[int, Any]], int]:
    model_data = torch.load(os.path.join(model_dir, "best"), map_location="cpu")

    config = read_config(os.path.join(model_dir, "config.json"))

    use_cuda = torch.cuda.is_available() and not no_cuda
    print("Using cuda", use_cuda)

    info = read_info(os.path.join(model_dir, "info.json"))

    model = PredictionModel(config, info, use_cuda=use_cuda).to(
        device_from_config(use_cuda=use_cuda)
    )
    model.load_state_dict(model_data, strict=False)

    train_data: Any
    train_data = [], [], []

    for name, a_labeler in labelers.items():
        a, b, c = a_labeler.get_label_data()
        train_data[0].extend(list(a))
        train_data[1].extend(list(b))
        train_data[2].extend(list(c))

    train_data = tuple([np.array(a) for a in train_data])

    loaded_data = StrideDataset(
        os.path.join(info["extract_dir"], "extract.db"),
        os.path.join(info["extract_dir"], "ontology.db"),
        os.path.join(model_dir, "info.json"),
        train_data,
        train_data,
    )

    results: Dict[str, Dict[int, Any]]

    results = {labeler_name: {} for labeler_name in labelers}

    final_transformation = partial(
        PredictionModel.finalize_data,
        info,
        config,
        device_from_config(use_cuda=use_cuda),
    )

    with dataset.BatchIterator(
        loaded_data,
        final_transformation,
        is_val=False,
        batch_size=2000,
        seed=0,
        day_dropout=0,
        code_dropout=0,
    ) as batches:
        for batch in batches:
            with torch.no_grad():
                embeddings = model.compute_embedding_batch(batch["rnn"]).cpu()
                for i, patient_id in enumerate(batch["pid"]):
                    for labeler_name, labeler in labelers.items():
                        labels = labeler.label(
                            patient=None, patient_id=patient_id
                        )
                        real_offsets = [label.day_index for label in labels]
                        try:
                            results[labeler_name][patient_id] = embeddings[
                                i, real_offsets, :
                            ].numpy()
                        except:
                            print(i, patient_id)
                            print(batch["pid"])
                            print(embeddings.shape)
                            print(labels)
                            raise

    return results, config["size"]
