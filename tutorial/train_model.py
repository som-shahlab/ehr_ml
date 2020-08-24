import shutil
import os

import scipy.sparse
import numpy as np
import json

import lightgbm

import sklearn.metrics

from typing import Any

def create_dataset(split: str) -> Any:
    with open(os.path.join(split, 'features.npz'), 'rb') as f:
        features = scipy.sparse.load_npz(f)

    with open(os.path.join(split, 'labels.npy'), 'rb') as f:
        labels = np.load(f)

    return lightgbm.Dataset(features, list(labels))

def get_matrix(split: str) -> Any:
    with open(os.path.join(split, 'features.npz'), 'rb') as f:
        return scipy.sparse.load_npz(f)

print('Start loading the data')

train_dataset = create_dataset('train')
dev_dataset = create_dataset('dev')
test_dataset = create_dataset('test')

print('Loading complete')


param = {
   'num_leaves':200,
   'objective':'binary',
   'metric': ['auc'],
   'num_threads': 3,
}

tree = lightgbm.train(param, train_dataset, valid_sets=[dev_dataset], early_stopping_rounds=10)

train_auroc = sklearn.metrics.roc_auc_score(train_dataset.get_label(), tree.predict(get_matrix('train')))
dev_auroc = sklearn.metrics.roc_auc_score(dev_dataset.get_label(), tree.predict(get_matrix('dev')))
test_auroc = sklearn.metrics.roc_auc_score(test_dataset.get_label(), tree.predict(get_matrix('test')))

print(f'Train {train_auroc}')
print(f'Dev {dev_auroc}')
print(f'Test {test_auroc}')
