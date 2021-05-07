import csv
import urllib.request
import argparse
import codecs
import collections
import gzip
import os


def clean_synpuf() -> None:
    parser = argparse.ArgumentParser(
        description="A tool for cleaning up a synpuf dataset for use with ehr_ml tools",
    )

    parser.add_argument(
        "synpuf_source_dir", type=str, help="Path of the source synpuf folder",
    )

    parser.add_argument(
        "synpuf_target_dir",
        type=str,
        help="Path to the target fixed synpuf folder",
    )

    args = parser.parse_args()

    table_fields = collections.defaultdict(list)

    with urllib.request.urlopen(
        r"https://raw.githubusercontent.com/OHDSI/CommonDataModel/v5.2.2/PostgreSQL/OMOP%20CDM%20ddl%20-%20PostgreSQL.sql"
    ) as f:
        uf = codecs.iterdecode(f, "utf-8", errors="ignore")

        current_table = None

        for line in uf:
            if line.strip() == ")":
                continue
            elif "CREATE TABLE" in line:
                parts = line.split()
                current_table = parts[2]
            elif line.strip() == "(":
                continue
            elif ";" in line:
                current_table = None
            elif current_table is not None:
                parts = line.split()
                table_fields[current_table].append(parts[0])

    headers = {a: ("\t".join(b) + "\n") for a, b in table_fields.items()}

    os.mkdir(args.synpuf_target_dir)

    for name in os.listdir(args.synpuf_source_dir):
        header = headers.get(name.lower().split(".")[0])
        if header is None:
            continue

        with open(os.path.join(args.synpuf_source_dir, name)) as i:
            with gzip.open(
                os.path.join(args.synpuf_target_dir, name.lower() + ".gz"), "wt"
            ) as o:
                first_line = i.readline()
                if first_line != header:
                    o.write(header)
                o.write(first_line)

                for l in i:
                    o.write(l)
