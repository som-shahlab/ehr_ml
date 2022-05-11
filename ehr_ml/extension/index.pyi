from __future__ import annotations

from typing import Iterable, Sequence

from . import timeline

def create_index(
    parent_timelines: str,
    output_filename: str,
) -> None: ...

class Index:
    def __init__(self, filename: str): ...
    def get_patient_ids(self, term: int) -> Sequence[int]: ...
    def get_all_patient_ids(self, terms: Iterable[int]) -> Sequence[int]: ...
