import ehr_ml.labeler
import ehr_ml.ontology
import ehr_ml.timeline
import ehr_ml.splits
import ehr_ml.labeler

import shutil
import os

from my_labeler import LongAdmissionLabeler

extract_path = '/labs/shahlab/datasets/ehr_ml_extracts/optum/0_0'

timelines = ehr_ml.timeline.TimelineReader(os.path.join(extract_path, 'extract.db'))
ind = ehr_ml.index.Index(os.path.join(extract_path, 'index.db'))
for split in ['train', 'dev', 'test']:
    shutil.rmtree(split, ignore_errors=True)
    os.mkdir(split)

    start, end, _ = ehr_ml.splits.read_time_split(extract_path, split)
    print('labeling for', split, start, end)

    subset = range(100000)

    labeler = ehr_ml.labeler.PatientSubsetLabeler(ehr_ml.labeler.PredictionAfterDateLabeler(LongAdmissionLabeler(timelines, ind), start_date=start), subset)
    ehr_ml.labeler.SavedLabeler.save(labeler, timelines, os.path.join(split, 'saved_labels.json'), end_date=end)
