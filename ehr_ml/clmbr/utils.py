import os
import sys
import json
import torch
import logging
import datetime

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def read_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as file:
        config = json.load(file)
        return config


def read_info(info_file: str) -> Dict[str, Any]:
    if not Path(os.path.join(info_file)).is_file():
        print("Fatal error: info.json not found.", file=sys.stderr)
        logging.info("Fatal error: info.json not found")
        exit(1)

    def date_from_str(x: str) -> Optional[datetime.date]:
        if x == "None":
            return None
        else:
            date_obj = datetime.datetime.strptime(x, "%Y-%m-%d").date()
            return date_obj

    with open(info_file, "rb") as file:
        info = json.load(file)
        info["valid_code_map"] = {
            int(code): int(idx) for code, idx in info["valid_code_map"].items()
        }
        info["code_counts"] = {
            int(code): int(idx) for code, idx in info["code_counts"].items()
        }
        for date_name in [
            "train_start_date",
            "train_end_date",
            "val_start_date",
            "val_end_date",
        ]:
            if date_name in info:
                info[date_name] = date_from_str(info[date_name])
        return info


def device_from_config(use_cuda: bool) -> torch.device:
    return torch.device("cuda:0" if use_cuda else "cpu")
