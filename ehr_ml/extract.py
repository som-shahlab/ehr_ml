from __future__ import annotations

import argparse
from lib2to3.pytree import convert
from pathlib import PurePosixPath
from sys import prefix
from typing import Sequence, Mapping, Tuple

# from .extension.index import create_index
# from .extension.ontology import create_ontology
# from .extension.timeline import create_timeline

import datetime
import abc
import multiprocessing
import os
import gzip
import numbers
import csv

from dataclasses import dataclass

@dataclass
class Event:
    date: datetime.date
    code: str
    value: str | float

class Converter(abc.ABC):
    def __init__(self):
        super().__init__()

    def get_person_field(self) -> str:
        return "person_id"

    @abc.abstractmethod
    def get_file_prefix(self) -> str:
        ...

    @abc.abstractmethod
    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        ...

class DemographicsConverter(Converter):
    def get_file_prefix(self) -> str:
        return "person"

    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        if row['birth_datetime']:
            birth = datetime.datetime.fromisoformat(row['birth_datetime'])
        else:
            year = 1900
            month = 1
            day = 1

            if row['year_of_birth']:
                year = int(row[4])
            
            if row['month_of_birth']:
                month = int(row[5])

            if row['day_of_birth']:
                day = int(row[6])

            birth = datetime.datetime(year=year, month=month, day=day)

        return [
            Event(birth, "birth", None),
        ] + [Event(birth, row[target], None)
            for target in ["gender_source_concept_id",
                "ethnicity_source_concept_id",
                "race_source_concept_id"]
        ]

class StandardConceptTableConverter(Converter):
    def __init__(self, prefix: str, date_field: str, concept_id_field: str):
        super().__init__()
        self.prefix = prefix
        self.date_field = date_field
        self.concept_id_field = concept_id_field

    def get_file_prefix(self) -> str:
        return self.prefix

    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        return [Event(datetime.datetime.fromisoformat(row[self.date_field]), row[self.concept_id_field], None)]

def run_converter(args: Tuple[str, str, Converter]) -> None: 
    source, target, converter = args
    os.makedirs(os.path.dirname(target), exist_ok=True)

    print('Running', source, target, converter)

    with gzip.open(source, 'rt') as f:
        reader = csv.DictReader(f)
        with gzip.open(target, 'wt') as o:
            writer = csv.DictWriter(o, fieldnames=['person_id', 'date', 'code', 'text_value', 'numeric_value'])
            writer.writeheader()
            for row in reader:
                lower_row = {a.lower():b for a, b in row.items()}
                for event in converter.get_events(lower_row):
                    writer.writerow({
                        'person_id': row[converter.get_person_field()],
                        'date': event.date,
                        'code': event.code,
                        'text_value': event.value if isinstance(event.value, str) else '',
                        'numeric_value': event.value if isinstance(event.value, numbers.Number) else '',
                    })
    print('Done with', source, target, converter)

def generate_csvs(omop_source: str, target_location: str, num_threads: int) -> None:
    pool = multiprocessing.Pool(num_threads)

    converters = [
        DemographicsConverter(),

    StandardConceptTableConverter(
        "drug_exposure", "drug_exposure_start_date", "drug_source_concept_id"),
    StandardConceptTableConverter("condition_occurrence",
                                         "condition_start_date",
                                         "condition_source_concept_id")
    ]

    os.makedirs(target_location, exist_ok=True)

    to_process = []

    for root, dirs, files in os.walk(omop_source):
        for name in files:
            full_path = PurePosixPath(root, name)
            relative_path = full_path.relative_to(omop_source)
            target_path = PurePosixPath(target_location) / relative_path
            matching_converters = [a for a in converters if a.get_file_prefix() in str(relative_path)]

            if len(matching_converters) > 1:
                print('Multiple converters matched?', full_path, matching_converters)
                print(1/0)
            elif len(matching_converters) == 0:
                print('Ignoring file', full_path)
            else:
                converter = matching_converters[0]
                print('Processing', full_path, converter)
                to_process.append((full_path, target_path, converter))
    
    done = pool.map(run_converter, to_process, chunksize=1)

    pool.close()
    pool.join()

def extract_omop_program() -> None:
    parser = argparse.ArgumentParser(
        description="An extraction tool for OMOP v5 sources"
    )

    parser.add_argument(
        "omop_source", type=str, help="Path of the folder to the omop source",
    )

    parser.add_argument(
        "umls_location", type=str, help="The location of the umls directory",
    )

    parser.add_argument(
        "gem_location", type=str, help="The location of the gem directory",
    )

    parser.add_argument(
        "rxnorm_location",
        type=str,
        help="The location of the rxnorm directory",
    )

    parser.add_argument(
        "target_location", type=str, help="The place to store the extract",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="The delimiter used in the raw OMOP source",
    )

    parser.add_argument(
        "--ignore_quotes",
        dest="use_quotes",
        action="store_false",
        help="Ignore quotes while parsing",
    )
    parser.add_argument(
        "--use_quotes",
        dest="use_quotes",
        action="store_true",
        help="Use quotes while parsing",
    )
    parser.set_defaults(use_quotes=True)

    args = parser.parse_args()

    print(args)

    generate_csvs(args.omop_source, 'temp_events', 3)

    print(1/0)

    extract_omop(
        args.omop_source,
        args.umls_location,
        args.gem_location,
        args.rxnorm_location,
        args.target_location,
        args.delimiter,
        args.use_quotes,
    )
