from __future__ import annotations

import threading
import queue
import math
import torch

from . import StrideDataset
from .rnn_model import PatientRNN
from .sequential_task import SequentialTask
from .labeler_task import LabelerTask
from .doctorai_task import DoctorAITask

from typing import Any, Dict

def finalize_data(batch: Dict[Any, Any], device: torch.device):
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
    dataset: StrideDataset,
    args: Any,
    out_queue: queue.Queue,
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

class DataLoader:
    def __init__(
        self,
        dataset: StrideDataset,
        threshold: int,
        is_val: bool = False,
        batch_size: int = 2000,
        seed: int = 0,
        day_dropout: float = 0,
        code_dropout: float = 0,
        use_cuda: bool = torch.cuda.is_available()
    ):
        self.batch_queue: queue.Queue[Any] = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.num_batches = dataset.num_train_batches(batch_size)

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
