from __future__ import annotations

import argparse
import bisect
import json
import os

from .extension.timeline import (
    ObservationWithValue,
    TimelineReader,
    Patient,
    PatientDay,
)

from .extension.ontology import (
    OntologyReader
)

__all__ = ["ObservationWithValue", "TimelineReader", "Patient", "PatientDay"]


def create_patient_data() -> None:
    parser = argparse.ArgumentParser(
        description="A tool for inspecting an ehr_ml extract"
    )

    parser.add_argument(
        "extract_dir",
        type=str,
        help="Path of the folder to the ehr_ml extraction",
    )

    parser.add_argument(
        "patient_id", type=int, help="The patient id to inspect",
    )

    args = parser.parse_args()

    source_file = os.path.join(args.extract_dir, "extract.db")
    timelines = TimelineReader(source_file)
    if args.patient_id is not None:
        patient_id = int(args.patient_id)
    else:
        patient_id = timelines.get_patient_ids()[0]

    location = bisect.bisect_left(timelines.get_patient_ids(), patient_id)
    original_patient_id = timelines.get_original_patient_ids()[location]

    if timelines.get_patient_ids()[location] != patient_id:
        print("Could not locate patient ?", patient_id)
        exit(-1)

    patient = timelines.get_patient(patient_id)

    print(f"Patient: {patient.patient_id}, (aka {original_patient_id})")

    def obs_with_value_to_str(obs_with_value: ObservationWithValue) -> str:
        code_text = timelines.get_dictionary().get_word(obs_with_value.code)
        if obs_with_value.is_text:
            value_text = timelines.get_value_dictionary().get_word(
                obs_with_value.text_value
            )
            return f'{code_text}-"{value_text}"'
        else:
            return f"{code_text}-{obs_with_value.numeric_value}"

    ontology = OntologyReader(os.path.join(args.extract_dir, "ontology.db"))

    with open('test_patient.json', 'w', encoding='utf-8') as f:
        patient_timeline = {}
        for i, day in enumerate(patient.days):
            patient_timeline[f'day {i}'] = {
                "date": str(day.date),
                "age in days": str(day.age),
                "observation": {
                    str(timelines.get_dictionary().get_word(a)) : str(ontology.get_text_description_dictionary().get_definition(a))
                    for a in day.observations
                },
                "observation with values": sorted([
                    obs_with_value_to_str(a) for a in day.observations_with_values]),
            }

        json.dump(patient_timeline, f, sort_keys=True, indent=4)
