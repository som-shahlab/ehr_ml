import itertools
import math
import random
import datetime

from ..timeline import TimelineReader

import torch
from torch import nn
from collections import defaultdict

import embedding_dot


class DoctorAITask(nn.Module):
    """
    This is paired with an encoder that outputs an encoding for each timestep.  
    This is the output (and loss) module for that encoder.  An example of an 
    encoder is PatientRNN in rnn_model.py.  
    """

    def __init__(self, config, data_config, info):
        super().__init__()

        self.config = config
        self.data_config = data_config
        self.info = info

        if config["simple_constant_baseline"]:
            self.output_code_embedding = nn.Embedding(
                len(data_config["leaf_code_map"]), 1
            )
        else:
            self.output_code_embedding = nn.Embedding(
                len(data_config["leaf_code_map"]), self.config["size"] + 1
            )

        self.output_code_weight = self.output_code_embedding.weight

    def forward(self, rnn_output, data):
        reshaped_rnn_output = rnn_output.view(-1, self.config["size"])
        bias_tensor = torch.ones(
            reshaped_rnn_output.shape[0], 1, device=reshaped_rnn_output.device
        )

        if self.config["simple_constant_baseline"]:
            rnn_with_bias = bias_tensor
        else:
            rnn_with_bias = torch.cat([reshaped_rnn_output, bias_tensor], dim=1)

        non_text_output_weights, non_text_expected_output = data
        # We use the "boring" flat decoder
        intermediate = torch.mm(rnn_with_bias, self.output_code_weight.t())
        final = intermediate.view(rnn_output.shape[0], rnn_output.shape[1], -1)

        probabilities = nn.functional.softmax(final, dim=2)
        # probabilities = torch.sigmoid(final)

        loss = nn.functional.binary_cross_entropy(
            probabilities,
            non_text_expected_output,
            weight=non_text_output_weights,
            reduction="sum",
        )

        return final, loss

    @classmethod
    def positive_codes_iterator(
        cls, data_config, info, ontologies, patient, mask_before=None
    ):
        # No time prediction window, so predict next day
        index = -1
        for day in patient.days:
            if info.get("use_terms") or len(day.observations) != 0:
                index += 1

            if mask_before is not None:
                this_date = datetime.date(
                    year=day.date.year, month=day.date.month, day=day.date.day
                )
                if this_date < mask_before:
                    continue

            if index >= 1:
                positive_codes = set()
                for code in day.observations:
                    if code in info["recorded_date_codes"]:
                        for subword in ontologies.get_subwords(code):
                            positive_codes.add(subword)

                yield (index - 1, positive_codes)

    @classmethod
    def compute_leaf_code_map(cls, data_config, info, ontologies):
        children_map = ontologies.get_children_map()
        valid_codes_to_predict = set()

        next_leaf_node_index = 0
        leaf_code_map = {}

        for code in info["valid_code_map"].keys():
            word = ontologies.get_dictionary().get_word(int(code))
            code_type = word.split("/")[0]

            if code_type in {"ATC"}:
                children = children_map[code]
                valid_children = set(children) & set(
                    info["valid_code_map"].keys()
                )
                if len(valid_children) == 0:
                    leaf_code_map[code] = next_leaf_node_index
                    next_leaf_node_index += 1

            if code_type in {"ICD10CM"}:
                code_part = word.split("/")[1]
                if len(code_part) == 3:
                    leaf_code_map[code] = next_leaf_node_index
                    next_leaf_node_index += 1

        return leaf_code_map

    @classmethod
    def convert_samples_to_variables(
        cls, data_config, info, ontologies, patients, mask_before=None
    ):
        """
        Name is confusing but this constructs _targets_ from next day codes and terms. 
        """

        maximum_length = max(
            sum(
                info.get("use_terms") or len(day.observations) > 0
                for day in x.days
            )
            for x in patients
        )

        non_text_expected_output = torch.ByteTensor(
            len(patients), maximum_length, len(data_config["leaf_code_map"])
        ).fill_(0)
        non_text_output_weights = torch.ByteTensor(
            len(patients), maximum_length, len(data_config["leaf_code_map"])
        ).fill_(0)

        for i, patient in enumerate(patients):
            for j, positive_codes in cls.positive_codes_iterator(
                data_config, info, ontologies, patient, mask_before=mask_before
            ):
                any_added = False

                for positive_code in positive_codes:
                    index = data_config["leaf_code_map"].get(positive_code)
                    if index is not None:
                        non_text_expected_output[i, j, index] = 1
                        any_added = True

                if any_added:
                    non_text_output_weights[i, j, :] = 1

        return (non_text_output_weights, non_text_expected_output)

    @classmethod
    def finalize_data(cls, initial):
        non_text_output_weights, non_text_expected_output = initial
        non_text_output_weights = non_text_output_weights.to(torch.float)
        non_text_expected_output = non_text_expected_output.to(torch.float)
        return (non_text_output_weights, non_text_expected_output)
