from __future__ import annotations

import queue
import threading
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from . import StrideDataset


def prepare_batch_thread(
    dataset: StrideDataset,
    args: Any,
    out_queue: queue.Queue[Any],
    stop_event: threading.Event,
    transform_func: Any,
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

        result = transform_func(item)
        out_queue.put(result)


class BatchIterator:
    def __init__(
        self,
        dataset: StrideDataset,
        transform_func: Any,
        is_val: bool = False,
        batch_size: int = 2000,
        seed: int = 0,
        day_dropout: float = 0,
        code_dropout: float = 0,
    ):
        self.batch_queue: queue.Queue[Any] = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()

        args = (is_val, batch_size, seed, day_dropout, code_dropout)

        self.data_thread = threading.Thread(
            target=prepare_batch_thread,
            args=(
                dataset,
                args,
                self.batch_queue,
                self.stop_event,
                transform_func,
            ),
        )
        self.data_thread.start()
        self.stopped = False

    def __iter__(self) -> BatchIterator:
        return self

    def __enter__(self) -> BatchIterator:
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
