from __future__ import annotations

import os
import math
import queue
import torch
import bisect
import datetime
import threading
import numpy as np

from .. import timeline

from . import PatientTimelineDataset
from .rnn_model import PatientRNN
from .sequential_task import SequentialTask
from .labeler_task import LabelerTask
from .doctorai_task import DoctorAITask

from typing import Any, Dict, Optional, Iterable, Tuple, List

def finalize_data(batch: Dict[Any, Any], device: torch.device) -> Dict[Any, Any]:
    batch["pid"] = batch["pid"].tolist()
    batch["day_index"] = batch["day_index"].tolist()
    batch["rnn"] = PatientRNN.finalize_data(batch["rnn"], device)
    if "task" in batch:
        batch["task"] = SequentialTask.finalize_data(batch["task"], device)
    if "doctorai" in batch:
        batch["doctorai"] = DoctorAITask.finalize_data(batch["doctorai"])
    if "labeler" in batch:
        batch["labeler"] = LabelerTask.finalize_data(batch["labeler"])
    return batch

def prepare_batch_thread(
    dataset: PatientTimelineDataset,
    args: Any,
        out_queue: queue.Queue[Optional[Dict[Any, Any]]],
    stop_event: threading.Event,
    device: torch.device
) -> None:
    iterator = dataset.get_iterator(*args)
    while True:
        if stop_event.is_set():
            out_queue.put(None)
            break

        item = next(iterator, None)
        if item is None:
            out_queue.put(None)
            break

        result = finalize_data(item, device)
        out_queue.put(result)
        
def convert_patient_data(extract_dir: str,
                         original_patient_ids: Iterable[int],
                         date_strs: Iterable[str]) -> Tuple[np.array, np.array]:
    timelines = timeline.TimelineReader(os.path.join(extract_dir, "extract.db"))

    all_original_pids = timelines.get_original_patient_ids()
    all_ehr_ml_pids = timelines.get_patient_ids()

    def get_date_index(pid: int, date_obj: datetime.date) -> int:
        patient = timelines.get_patient(pid)
        for i, day in enumerate(patient.days):
            if date_obj == day.date:
                return i
        assert 0, "should find correct date in timeline!"

    def convert_data(og_pid: int, date_str: str) -> Tuple[int, int]:
        pid_index = bisect.bisect_left(all_original_pids, og_pid)
        assert (
            all_original_pids[pid_index] == og_pid
        ), f"original patient ID {og_pid} not in timeline"
        ehr_ml_pid = all_ehr_ml_pids[pid_index]

        date_obj = datetime.date.fromisoformat(date_str)
        date_index = get_date_index(ehr_ml_pid, date_obj)
        return ehr_ml_pid, date_index

    ehr_ml_patient_ids = []
    day_indices = []
    for og_pid, date_str in zip(original_patient_ids, date_strs):
        ehr_ml_pid, date_index = convert_data(og_pid, date_str)
        ehr_ml_patient_ids.append(ehr_ml_pid)
        day_indices.append(date_index)

    return np.array(ehr_ml_patient_ids), np.array(day_indices)

class DataLoader:
    def __init__(
        self,
        dataset: PatientTimelineDataset,
        threshold: int,
        is_val: bool = False,
        batch_size: int = 2000,
        seed: int = 0,
        day_dropout: float = 0,
        code_dropout: float = 0,
        use_cuda: Optional[bool] = None
    ):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.batch_queue: queue.Queue[Any] = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.num_batches = dataset.num_batches(batch_size, is_val)

        args = (is_val, batch_size, seed, threshold, day_dropout, code_dropout)

        self.data_thread = threading.Thread(
            target=prepare_batch_thread,
            args=(
                dataset,
                args,
                self.batch_queue,
                self.stop_event,
                torch.device("cuda:0" if use_cuda else "cpu")
            ),
        )
        self.data_thread.start()
        self.stopped = False

    def __len__(self) -> int:
        return self.num_batches
        
    def __iter__(self) -> DataLoader:
        return self

    def __enter__(self) -> DataLoader:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.stop_event.set()

        while not self.stopped:
            item = self.batch_queue.get()

            if item is None:
                self.stopped = True

        self.data_thread.join()

    def __next__(self) -> Any:
        next_item = self.batch_queue.get()
        if next_item is None:
            self.stopped = True
            raise StopIteration
        else:
            return next_item
