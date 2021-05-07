from __future__ import annotations

import datetime
from typing import Iterator, Literal, Optional, Sequence, Union

class TimelineReader:
    def __init__(self, filename: str, readall: bool = ...): ...
    def get_patient(
        self,
        patient_id: int,
        start_date: Optional[datetime.date] = ...,
        end_date: Optional[datetime.date] = ...,
    ) -> Patient: ...
    def get_patients(
        self,
        patient_ids: Optional[Sequence[int]] = ...,
        start_date: Optional[datetime.date] = ...,
        end_date: Optional[datetime.date] = ...,
    ) -> Iterator[Patient]: ...
    def get_patient_ids(self) -> Sequence[int]: ...
    def get_original_patient_ids(self) -> Sequence[int]: ...
    def get_dictionary(self) -> TermDictionary: ...
    def get_value_dictionary(self) -> TermDictionary: ...

class TermDictionary:
    def map(self, term: str) -> Optional[int]: ...
    def get_word(self, code: int) -> Optional[str]: ...
    def get_items(self) -> List[Tuple[str, int]]: ...

class Patient:
    patient_id: int
    days: Sequence[PatientDay]

class PatientDay:
    date: datetime.date
    age: int
    observations: Sequence[int]
    observations_with_values: Sequence[ObservationWithValue]

class NumericObservationWithValue:
    code: int
    is_text: Literal[False]
    numeric_value: float

class TextObservationWithValue:
    code: int
    is_text: Literal[True]
    text_value: int

ObservationWithValue = Union[
    NumericObservationWithValue, TextObservationWithValue
]
