import datetime
from typing import Any, Iterator

def create_info(
    timeline_path: str,
    ontology_path: str,
    train_end_date: datetime.date,
    val_end_date: datetime.date,
    min_patient_count: int,
) -> str: ...

class PatientTimelineDataset:
    def __init__(
        self, timelines_path: str, ontology_path: str, info_path: str
    ): ...
    def num_batches(self, batch_size: int, is_val: bool = ...) -> int: ...
    def get_iterator(
        self,
        is_val: bool,
        batch_size: int,
        seed: int,
        threshold: int,
        day_dropout: float = ...,
        code_dropout: float = ...,
    ) -> Iterator[Any]: ...
