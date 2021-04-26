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

from . import dataset
from .prediction_model import PredictionModel
from ..featurizer import ColumnValue, Featurizer
from ..splits import read_time_split
from ..utils import OnlineStatistics, set_up_logging

from .opt import OpenAIAdam

from typing import Mapping, Any, Dict, Optional, Tuple

def check_dir_for_overwrite(dirname: str) -> bool: 
    return bool(glob.glob(os.path.join(dirname, '*.json')) or glob.glob(os.path.join(dirname, 'checkpoints')))

def create_info_program() -> None:
    parser = argparse.ArgumentParser(
        description='Precompute training data summary statistics etc for CLMBR experiments')
    parser.add_argument('--extract_dir',
                        type=str, 
                        help='Extract dir')
    parser.add_argument('--train_end_date', 
                        type=str,
                        help='The end date for training')
    parser.add_argument('--val_end_date', 
                        type=str, 
                        help='The end date for validation')
    parser.add_argument('--min_patient_count', 
                        type=int, 
                        default=100, 
                        help='Only keep statistics on codes/terms that appear for this many patients')
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Override dir where model is saved')
    parser.add_argument('--banned_patient_file', 
                        type=str, 
                        help='A file containing a list of patients to exclude from training')
    args = parser.parse_args()

    if args.save_dir is None: 
        print('Error - must specify save_dir', file=sys.stderr)
        exit(1)
    else: 
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    set_up_logging(os.path.join(save_dir, 'create_info.log'))
    logging.info('Args: %s', str(args))

    if check_dir_for_overwrite(save_dir) : 
        print("Fatal error - model dir {} is not empty".format(save_dir), file=sys.stderr)
        logging.info("Fatal error - model dir {} is not empty".format(save_dir))
        exit(1)

    ontologies_path = os.path.join(args.extract_dir, 'ontology.db')
    timelines_path = os.path.join(args.extract_dir, 'extract.db')

    train_end_date = datetime.datetime.fromisoformat(args.train_end_date)
    val_end_date = datetime.datetime.fromisoformat(args.val_end_date)

    result = json.loads(create_info(timelines_path, ontologies_path, train_end_date, val_end_date, args.min_patient_count))
    result['extract_dir'] = args.extract_dir
    result['extract_file'] = 'extract.db'
    result['train_start_date'] = '1900-01-01'
    result['train_end_date'] = args.train_end_date
    result['val_start_date'] = args.train_end_date
    result['val_end_date'] = args.val_end_date
    result['seed'] = 3451235
    result['min_patient_count'] = args.min_patient_count

    if args.banned_patient_file:
        with open(args.banned_patient_file) as f:
            pids = {int(a) for a in f}

            def remove_banned(a):
                return [(p, c) for p, c in a if p not in pids]

            result["train_patient_ids_with_length"] = remove_banned(result["train_patient_ids_with_length"])
            result["val_patient_ids_with_length"] = remove_banned(result["val_patient_ids_with_length"])

    def count_frequent_items(counts: Mapping[Any, int], threshold: int) -> int: 
        return len({ item for item, count in counts.items() if count >= threshold})

    logging.info('Codes with >= 10 {}'.format(count_frequent_items(result['code_counts'], 10)))
    logging.info('Codes with >= 25 {}'.format(count_frequent_items(result['code_counts'], 25)))
    logging.info('Codes with >= 50 {}'.format(count_frequent_items(result['code_counts'], 50)))
    logging.info('Codes with >= 100 {}'.format(count_frequent_items(result['code_counts'], 100)))
    logging.info('Codes with >= 1000 {}'.format(count_frequent_items(result['code_counts'], 1000)))
    logging.info('Number codes: {}'.format(len(result['code_counts'])))
    logging.info('Number valid codes: {}'.format(len(result['valid_code_map'])))

    with open(os.path.join(args.save_dir, 'info.json'), 'w') as fp:
        json.dump(result, fp)

def read_config(config_file: str) -> Dict[str, Any]: 
    with open(config_file, "r") as file: 
        config = json.load(file)
        return config

def read_info(info_file: str) -> Dict[str, Any]: 
    if not Path(os.path.join(info_file)).is_file(): 
        print('Fatal error: info.json not found.', file=sys.stderr)
        logging.info('Fatal error: info.json not found')
        exit(1)

    def date_from_str(x: str) -> Optional[datetime.date]: 
        if x == 'None' : 
            return None
        else : 
            date_obj = datetime.datetime.strptime(x, '%Y-%m-%d').date()
            return date_obj

    with open(info_file, 'rb') as file:
        info = json.load(file)
        info['valid_code_map'] = { int(code) : int(idx) for code, idx in info['valid_code_map'].items() }
        info['code_counts'] = { int(code) : int(idx) for code, idx in info['code_counts'].items() }
        for date_name in ['train_start_date', 'train_end_date', 'val_start_date', 'val_end_date']:
            if date_name in info:
                info[date_name] = date_from_str(info[date_name])
        return info

def device_from_config(use_cuda: bool) -> torch.device: 
    return torch.device("cuda:0" if use_cuda else "cpu")

def train_model() -> None:
    parser = argparse.ArgumentParser(
        description='Representation Learning Experiments')
    parser.add_argument('--model_dir', 
                        type=str, 
                        default=None, 
                        help='Override where model is saved')
    parser.add_argument('--info_dir',
                        type=str,
                        default=None)
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.01,
                        help='learning rate')
    parser.add_argument('--size', 
                        default=768,
                        type=int,)
    parser.add_argument('--use_gru', 
                        default=False,
                        action='store_true', )
    parser.add_argument('--no_tied_weights', 
                        default=False,
                        action='store_true')
    parser.add_argument('--gru_layers', 
                        default=1,
                        type=int)
    parser.add_argument('--gru_hidden_size', 
                        default=768,
                        type=int)
    parser.add_argument('--dropout', 
                        default=0,
                        type=float)
    parser.add_argument('--l2', 
                        default=0.01,
                        type=float)
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=500, 
                        help='Batch size')
    parser.add_argument('--extract_dir', 
                        action='store_true',
                        help="Use the doctorai task definition")
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--code_dropout', 
                        type=float, 
                        default=0.2)
    args = parser.parse_args()

    if args.info_dir is None or not Path(args.info_dir).is_dir(): 
        print("Error - must provide path to info directory", file=sys.stderr)
        exit(1)

    if args.model_dir is None: 
        print("Error - must provide model dir", file=sys.stderr)
        exit(1)

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    
    epochs_per_cycle = 50
    warmup_epochs = 2

    # Try to load info.json file; see create_info above for details. 
    info = read_info(os.path.join(args.info_dir, 'info.json'))
    copyfile(os.path.join(args.info_dir, 'info.json'), os.path.join(model_dir, 'info.json'))

    first_too_small_index = float('inf')
    for code, index in info['valid_code_map'].items():
        if info['code_counts'][code] < 10 * info['min_patient_count']:
            first_too_small_index = min(first_too_small_index, index)

    print(len(info['valid_code_map']), flush=True)

    # Create and save config dictionary
    config = {
        'batch_size': args.batch_size,
        'num_first': first_too_small_index,
        'num_second': len(info['valid_code_map']) - first_too_small_index,

        'size': args.size,

        'lr': args.lr, 
        'dropout': args.dropout,

        'use_gru': args.use_gru,
        'gru_layers': args.gru_layers,
        'gru_hidden_size': args.gru_hidden_size,
        'tied_weights': not args.no_tied_weights,

        'l2': args.l2,
        'b1': 0.9,
        'b2': 0.999,
        'e': 1e-8,

        'epochs_per_cycle': epochs_per_cycle,
        'warmup_epochs': warmup_epochs,
        'code_dropout': args.code_dropout,
    }

    with open(os.path.join(model_dir, 'config.json'), 'w') as outfile: 
        json.dump(config, outfile)

    set_up_logging(os.path.join(model_dir, 'train.log'))
    logging.info('Args: %s', str(args))

    loaded_data = StrideDataset( 
        os.path.join(info['extract_dir'], 'extract.db'),
        os.path.join(info['extract_dir'], 'ontology.db'),
        os.path.join(args.info_dir, 'info.json'))

    random.seed(info['seed'])

    use_cuda = torch.cuda.is_available() and not args.no_cuda

    model = PredictionModel(config, info, use_cuda=use_cuda).to(device_from_config(use_cuda=use_cuda))
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)

    final_transformation = partial(PredictionModel.finalize_data, config, info, device_from_config(use_cuda=use_cuda))

    print('Iters per epoch', loaded_data.num_train_batches(config['batch_size']))

    optimizer = OpenAIAdam(params, lr=config['lr'], schedule='warmup_linear', warmup=config['warmup_epochs']/config['epochs_per_cycle'], 
            t_total=loaded_data.num_train_batches(config['batch_size']) * config['epochs_per_cycle'], b1=config['b1'], b2=config['b2'], e=config['e'], l2=config['l2'])

    def train_epoch() -> None:
        model.train()

        all_non_text_loss = 0

        last_time = time.time()

        code_dropout = config['code_dropout']
        day_dropout = config['code_dropout']
        print(f'Code dropout is {code_dropout}')
        print(f'Day dropout is {day_dropout}')

        with dataset.BatchIterator(loaded_data, final_transformation, threshold=config['num_first'], is_val=False, batch_size=config['batch_size'], seed=random.randint(0, 100000), day_dropout=day_dropout, code_dropout=code_dropout) as batches:
            for i, batch in enumerate(batches):
                values, non_text_loss = model(batch)
                del values

                optimizer.zero_grad()
                non_text_loss.backward()
                optimizer.step()

                del non_text_loss
                del batch

                if i % 2000 == 0:
                    current_time = time.time()
                    delta = current_time - last_time
                    if i != 0:
                        print('Iters per second ', 2000 / delta, ' ' , i)
                    last_time = current_time




    def test(sample_size: int = 100) -> Tuple[float, float]:
        model.eval()
        train_loss = test_helper(is_val=False, sample_size=sample_size)
        val_loss = test_helper(is_val=True, sample_size=sample_size)
        return train_loss, val_loss

    def test_helper(is_val: bool, sample_size: int) -> float:
        non_text_total_loss = 0

        with dataset.BatchIterator(loaded_data, final_transformation, threshold=config['num_first'], is_val=is_val, batch_size=2000, seed=0, day_dropout=0, code_dropout=0) as batches:
            for batch, _ in zip(batches, range(sample_size)):
                with torch.no_grad():
                    values, non_text_loss = model(batch)
                    del values

                    non_text_total_loss += non_text_loss.item()

                    del batch
                    del non_text_loss

        return non_text_total_loss / sample_size

    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = None
    best_val_loss_epoch = None

    logging.info('Start training (v2)')
    with open(os.path.join(model_dir, 'losses'), 'w') as loss_file:
        for epoch in range(config['epochs_per_cycle']):
            logging.info('About to start epoch %s', epoch)
            train_epoch()
            logging.info('Epoch is complete %s', epoch)

            train_loss, val_loss = test(sample_size=2000)

            logging.info('Train loss: %s', train_loss)
            logging.info('Val loss: %s', val_loss)

            loss_file.write('Epoch {}\n'.format(epoch))
            loss_file.write('Train loss {}\n'.format(train_loss))
            loss_file.write('Val loss {}\n'.format(val_loss))
            loss_file.write('\n')
            loss_file.flush()

            # if epoch == 49: 
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch

                if os.path.exists(os.path.join(model_dir, 'best')):
                    os.unlink(os.path.join(model_dir, 'best'))

                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'best'))


def featurize_patients(model_dir: str, extract_dir: str) -> Tuple[np.array, np.array, np.array]:
    """
    Featurize patients using the given model.
    This function will use the GPU if it is available.
    """

    config = read_config(os.path.join(model_dir, "config.json"))
    info = read_info(os.path.join(model_dir, "info.json"))

    use_cuda = torch.cuda.is_available()

    model = PredictionModel(config, info, use_cuda=use_cuda).to(
        device_from_config(use_cuda=use_cuda)
    )
    model_data = torch.load(os.path.join(model_dir, "best"), map_location="cpu")
    model.load_state_dict(model_data)

    final_transformation = partial(
        PredictionModel.finalize_data,
        config,
        info,
        device_from_config(use_cuda=use_cuda),
    )

    loaded_data = StrideDataset(
        os.path.join(extract_dir, "extract.db"),
        os.path.join(extract_dir, "ontology.db"),
        os.path.join(model_dir, "info.json"),
    )

    patient_ids = []
    patient_indices = []
    patient_id_to_info = defaultdict(dict)
    patient_day_idx = 0
    
    # set up data iterator for collecting patient stats
    is_val = True
    batch_size = 10000
    seed = info['seed']
    threshold = config['num_first']
    day_dropout = 0
    code_dropout = 0
    iterator_args = (is_val, batch_size, seed, threshold, day_dropout, code_dropout)
    for item in loaded_data.get_iterator(*iterator_args):
        for pid in item['pid']:
            for index in range(item['day_index'].shape[-1]):
                patient_ids.append(pid)
                patient_indices.append(patient_day_idx)
                patient_id_to_info[pid][index] = patient_day_idx
                patient_day_idx += 1

    patient_representations = np.zeros((patient_day_idx + 1, config['size']))
    print(f'Total # patient days = {patient_day_idx + 1}', flush=True)
    print(f'Total # batches = {(patient_day_idx + 1) / 10000}', flush=True)

    with dataset.BatchIterator(loaded_data, final_transformation, threshold=threshold, is_val=is_val, batch_size=batch_size, seed=seed, day_dropout=day_dropout, code_dropout=code_dropout) as batches:
        pbar = tqdm(batches)
        pbar.set_description('Computing patient representations')
        for batch in pbar:
             with torch.no_grad():
                embeddings = (
                    model.compute_embedding_batch(batch["rnn"]).cpu().numpy()
                )
                for i, patient_id in enumerate(batch["pid"]):
                    for index, target_id in patient_id_to_info[patient_id].items():
                        patient_representations[target_id, :] = embeddings[
                            i, index, :
                        ]

    return patient_representations, np.array(patient_ids), np.array(patient_indices)

def featurize_patients_w_labels(model_dir: str, extract_dir: str, l: labeler.SavedLabeler) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Featurize patients using the given model and labeler.
    The result is a numpy array aligned with l.get_labeler_data().
    This function will use the GPU if it is available.
    """

    config = read_config(os.path.join(model_dir, "config.json"))
    info = read_info(os.path.join(model_dir, "info.json"))

    use_cuda = torch.cuda.is_available()

    model = PredictionModel(config, info, use_cuda=use_cuda).to(
        device_from_config(use_cuda=use_cuda)
    )
    model_data = torch.load(os.path.join(model_dir, "best"), map_location="cpu")
    model.load_state_dict(model_data)

    final_transformation = partial(
        PredictionModel.finalize_data,
        config,
        info,
        device_from_config(use_cuda=use_cuda),
    )

    data = l.get_label_data()

    labels, patient_ids, patient_indices = data
    
    loaded_data = StrideDataset(
        os.path.join(extract_dir, "extract.db"),
        os.path.join(extract_dir, "ontology.db"),
        os.path.join(model_dir, "info.json"),
        data,
        data,
    )

    patient_id_to_info = defaultdict(dict)
    is_val = True
    batch_size = 10000
    seed = info['seed']
    threshold = config['num_first']
    day_dropout = 0
    code_dropout = 0

    for i, (pid, index) in enumerate(zip(patient_ids, patient_indices)):
        patient_id_to_info[pid][index] = i

    patient_representations = np.zeros((len(labels), config['size']))
    print(f'Total # patient days = {len(labels)}', flush=True)
    print(f'Total # batches = {len(labels) / batch_size}', flush=True)

    with dataset.BatchIterator(loaded_data, final_transformation, threshold=threshold, is_val=is_val, batch_size=batch_size, seed=seed, day_dropout=day_dropout, code_dropout=code_dropout) as batches:
        pbar = tqdm(batches)
        pbar.set_description('Computing patient representations')
        for batch in pbar:
             with torch.no_grad():
                embeddings = (
                    model.compute_embedding_batch(batch["rnn"]).cpu().numpy()
                )
                for i, patient_id in enumerate(batch["pid"]):
                    for index, target_id in patient_id_to_info[patient_id].items():
                        patient_representations[target_id, :] = embeddings[
                            i, index, :
                        ]

    return patient_representations, labels, patient_ids, patient_indices
    


def debug_model() -> None:
    parser = argparse.ArgumentParser(
        description='Representation Learning Experiments')
    parser.add_argument('--model_dir', 
                        type=str, 
                        help='Override where model is saved')
    args = parser.parse_args()

    model_dir = args.model_dir

    config = read_config(os.path.join(model_dir, 'config.json'))
    info = read_info(os.path.join(model_dir, 'info.json'))
    use_cuda = torch.cuda.is_available()

    model = PredictionModel(config, info, use_cuda=use_cuda).to(device_from_config(use_cuda=use_cuda))
    model_data = torch.load(
        os.path.join(model_dir, 'best'), map_location='cpu')
    model.load_state_dict(model_data)

    loaded_data = StrideDataset( 
        os.path.join(info['extract_dir'], 'extract.db'),
        os.path.join(info['extract_dir'], 'ontology.db'),
        os.path.join(model_dir, 'info.json'))

    final_transformation = partial(PredictionModel.finalize_data, config, info, device_from_config(use_cuda=use_cuda))

    ontologies = ontology.OntologyReader(os.path.join(info['extract_dir'], 'ontology.db'))
    timelines = timeline.TimelineReader(os.path.join(info['extract_dir'], 'extract.db'))

    reverse_map = {}
    for b, a in info['valid_code_map'].items():
        word = ontologies.get_dictionary().get_word(b)
        reverse_map[a] = word

    reverse_map[len(info['valid_code_map'])] = 'None'

    with dataset.BatchIterator(loaded_data, final_transformation, threshold=config['num_first'], is_val=True, batch_size=1, seed=info['seed'], day_dropout=0, code_dropout=0) as batches:
        for batch in batches:
            if batch['task'][0].size()[0] == 0:
                continue
            values, non_text_loss = model(batch)
            values = torch.sigmoid(values)

            patient_id = int(batch['pid'][0])
            patient = timelines.get_patient(patient_id)
            original_day_indices = batch['day_index'][0]

            indices, targets, seen_before, _, _, _ = batch['task']
            day_indices = indices[:, 0]
            word_indices = indices[:, 1]

            all_non_text_codes, all_non_text_offsets, all_non_text_codes1, all_non_text_offsets1, all_day_information, all_positional_encoding, all_lengths = batch['rnn']

            all_non_text_codes = list(all_non_text_codes)
            all_non_text_offsets = list(all_non_text_offsets) + [len(all_non_text_codes)]

            print(patient_id, batch['pid'], original_day_indices)

            all_seen = set()

            for i, index in enumerate(original_day_indices):
                day = patient.days[index]
                print('------------------')
                print(patient_id, i, index, day.age/365, day.date)

                words = set()
                for code in day.observations:
                    for subword in ontologies.get_subwords(code):
                        words.add(ontologies.get_dictionary().get_word(subword))
                        all_seen.add(ontologies.get_dictionary().get_word(subword))

                print('Source', words)

                wordsA = set()
                
                if (i + 1) < len(all_non_text_offsets):
                    for code in all_non_text_codes[all_non_text_offsets[i]: all_non_text_offsets[i+1]]:
                        wordsA.add(reverse_map[code.item()])

                print('Given', wordsA)


                day_mask = (day_indices == i)

                w = word_indices[day_mask]
                p = values[day_mask]
                t = targets[day_mask]
                f = seen_before[day_mask]

                items = [(t_i.item(), reverse_map[w_i.item()], p_i.item(), reverse_map[w_i.item()] in all_seen, w_i.item(), f_i.item()) for p_i, t_i, w_i, f_i in zip(p, t, w, f)]
                items.sort(key = lambda x: (-x[0], x[1]))

                for a in items:
                    print(a)
