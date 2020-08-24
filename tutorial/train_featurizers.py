import ehr_ml.featurizer
import ehr_ml.timeline
import ehr_ml.labeler
import ehr_ml.splits

import json
import os

extract_path = '/labs/shahlab/datasets/ehr_ml_extracts/optum/0_0'
split = 'train'

timelines = ehr_ml.timeline.TimelineReader(os.path.join(extract_path, 'extract.db'))
ontologies = ehr_ml.ontology.OntologyReader(os.path.join(extract_path, 'ontology.db'))
start, end, _ = ehr_ml.splits.read_time_split(extract_path, split)
print('featurizing for', split, start, end)

with open(os.path.join(split, 'saved_labels.json')) as fp:
    labeler = ehr_ml.labeler.SavedLabeler(fp)

featurizers = ehr_ml.featurizer.FeaturizerList(
    [ehr_ml.featurizer.AgeFeaturizer(normalize=False), ehr_ml.featurizer.CountFeaturizer(timelines, ontologies)])

featurizers.train_featurizers(timelines, labeler, end_date=end)

with open(os.path.join(split, 'saved_featurizers.json'), 'w') as fp:
    featurizers.save(fp)