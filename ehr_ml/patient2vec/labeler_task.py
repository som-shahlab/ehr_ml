import datetime
import itertools
import math
import random
from collections import defaultdict, deque
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn

from ..timeline import TimelineReader


class LabelerTask(nn.Module):
    """
    This is paired with an encoder that outputs an encoding for each timestep.  
    This is the output (and loss) module for that encoder.  An example of an 
    encoder is PatientRNN in rnn_model.py.  
    """

    def __init__(self, config: Mapping[str, Any], info: Mapping[str, Any]):
        super().__init__()

        self.config = config
        self.info = info

        self.final_layer = nn.Linear(config["size"], 1, bias=True)

    def forward(self, rnn_output: torch.Tensor, data: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
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
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        initial: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        indices, targets = initial
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.float, device=device)
        return indices, targets
