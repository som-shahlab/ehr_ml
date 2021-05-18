from ..timeline import TimelineReader

import torch
import torch.nn.functional as F
from torch import nn

import embedding_dot


class SequentialTask(nn.Module):
    """
    This is paired with an encoder that outputs an encoding for each timestep.  
    This is the output (and loss) module for that encoder.  An example of an 
    encoder is PatientRNN in rnn_model.py.  
    """

    def __init__(self, config, info, weight, weight1):
        super().__init__()

        self.config = config
        self.info = info

        self.output_code_weight = weight
        self.output_code_weight1 = weight1

    def forward(self, rnn_output, data):
        reshaped_rnn_output = rnn_output.view(-1, self.config["size"])
        bias_tensor = torch.ones(
            reshaped_rnn_output.shape[0], 1, device=reshaped_rnn_output.device
        )

        rnn_with_bias = torch.cat([reshaped_rnn_output, bias_tensor], dim=1)

        (
            non_text_indices,
            non_text_expected_output,
            seen_before,
            non_text_indices1,
            non_text_expected_output1,
            seen_before1,
        ) = data

        final = embedding_dot.embedding_dot(
            rnn_with_bias, self.output_code_weight, non_text_indices
        )

        loss = F.binary_cross_entropy_with_logits(
            final, non_text_expected_output, reduction="sum"
        )

        small_size = (self.config["size"] // 4) + 1

        smaller = rnn_with_bias[:, -small_size:]

        final1 = embedding_dot.embedding_dot(
            rnn_with_bias[:, -small_size:].contiguous(),
            self.output_code_weight1,
            non_text_indices1,
        )

        loss1 = F.binary_cross_entropy_with_logits(
            final1, non_text_expected_output1, reduction="sum"
        )

        return final, (loss + loss1)

    @classmethod
    def finalize_data(cls, initial, device):
        (
            non_text_indices,
            non_text_expected_output,
            seen_before,
            non_text_indices1,
            non_text_expected_output1,
            seen_before1,
        ) = initial
        non_text_indices = torch.tensor(
            non_text_indices, device=device, dtype=torch.long
        )
        non_text_expected_output = torch.tensor(
            non_text_expected_output, device=device, dtype=torch.float
        )
        seen_before = torch.tensor(
            seen_before, device=device, dtype=torch.float
        )
        non_text_indices1 = torch.tensor(
            non_text_indices1, device=device, dtype=torch.long
        )
        non_text_expected_output1 = torch.tensor(
            non_text_expected_output1, device=device, dtype=torch.float
        )
        seen_before1 = torch.tensor(
            seen_before1, device=device, dtype=torch.float
        )
        return (
            non_text_indices,
            non_text_expected_output,
            seen_before,
            non_text_indices1,
            non_text_expected_output1,
            seen_before1,
        )
