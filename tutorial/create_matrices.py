import ehr_ml.labeler
import ehr_ml.ontology
import ehr_ml.timeline
import ehr_ml.splits
import ehr_ml.featurizer

import shutil
import os

import scipy.sparse
import numpy as np
import json

extract_path = '/labs/shahlab/datasets/ehr_ml_extracts/optum/0_0'

timelines = ehr_ml.timeline.TimelineReader(os.path.join(extract_path, 'extract.db'))
ontologies = ehr_ml.ontology.OntologyReader(os.path.join(extract_path, 'ontology.db'))

featurizers = ehr_ml.featurizer.FeaturizerList(
    [ehr_ml.featurizer.AgeFeaturizer(normalize=False), ehr_ml.featurizer.CountFeaturizer(timelines, ontologies)])

with open(os.path.join('train', 'saved_featurizers.json')) as fp:
    featurizers.load(fp)

for split in ['train', 'dev', 'test']:
    start, end, _ = ehr_ml.splits.read_time_split(extract_path, split)
    print('creating matrix for', split, start, end)

    with open(os.path.join(split, 'saved_labels.json')) as fp:
        labeler = ehr_ml.labeler.SavedLabeler(fp)

    features, labels, patient_ids, day_offsets = featurizers.featurize(timelines, labeler, end_date=end)

    with open(os.path.join(split, 'features.npz'), 'wb') as f:
        scipy.sparse.save_npz(f, features)

    with open(os.path.join(split, 'labels.npy'), 'wb') as f:
        np.save(f, labels)

    with open(os.path.join(split, 'patient_ids.npy'), 'wb') as f:
        np.save(f, patient_ids)

    with open(os.path.join(split, 'day_offsets.npy'), 'wb') as f:
        np.save(f, day_offsets)
