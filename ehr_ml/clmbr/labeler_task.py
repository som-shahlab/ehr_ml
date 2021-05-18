import itertools
import math
import random
import datetime

from ..timeline import TimelineReader

import torch
from torch import nn
from collections import defaultdict, deque


class LabelerTask(nn.Module):
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

        self.final_layer = nn.Linear(config["size"], 1, bias=True)

    def forward(self, rnn_output, data):
        indices, targets = data

        flat_rnn_output = rnn_output.view(-1, self.config["size"])

        correct_output = nn.functional.embedding(indices, flat_rnn_output)

        final = self.final_layer(correct_output)

        final = final.view((final.shape[0],))

        loss = nn.functional.binary_cross_entropy_with_logits(
            final, targets, reduction="sum"
        )

        return final, loss

    @classmethod
    def convert_samples_to_variables(
        cls, data_config, info, ontologies, patients, labeler
    ):
        """
        Name is confusing but this constructs _targets_ from next day codes and terms. 
        """

        max_length = 0
        indices = []
        targets = []

        for i, patient in enumerate(patients):
            labels = deque(labeler.label(patient))

            index = -1
            for day_offset, day in enumerate(patient.days):
                if len(labels) == 0:
                    break

                if len(day.observations) != 0:
                    index += 1

                if day_offset == labels[0].day_index:
                    indices.append((i, index))
                    targets.append(labels[0].is_positive)
                    labels.popleft()
                    max_length = max(max_length, index + 1)

        indices = torch.ShortTensor(
            [(i * max_length + index) for i, index in indices]
        )
        targets = torch.ByteTensor(targets)

        return (indices, targets)

    @classmethod
    def finalize_data(cls, initial):
        indices, targets = initial
        indices = indices.to(torch.long)
        targets = targets.to(torch.float)
        return indices, targets
