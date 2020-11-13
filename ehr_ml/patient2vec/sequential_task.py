import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import embedding_dot
import torch
from torch import nn

n = 20
points = torch.tensor([float(1 + i) for i in range(n)])
points = 20 * 365 * points / torch.sum(points)

end = torch.cumsum(points, dim=0)
start = end - points

print(start / 365, end / 365)

linear = True
print("Linear ", linear)

# from torch import autograd
# autograd.set_detect_anomaly(True)


class SequentialTask(nn.Module):
    """
    This is paired with an encoder that outputs an encoding for each timestep.  
    This is the output (and loss) module for that encoder.  An example of an 
    encoder is PatientRNN in rnn_model.py.  
    """

    def __init__(self, config: Mapping[str, Any], info: Mapping[str, Any]):
        super().__init__()
        self.config = config
        self.info = info

        # self.scale_and_scale_function = nn.Linear(config['size'], config['num_valid_targets'] * 3)
        # self.scale_and_scale_function = nn.Linear(config['size'], config['num_valid_targets'] * 2)
        # self.delta_function = nn.Linear(config['size'], config['num_valid_targets'] * 2 * n)

        self.main_weights = torch.nn.Embedding(
            config["num_valid_targets"], config["size"] + 1
        )

        self.sub_weights = torch.nn.Embedding(
            config["num_valid_targets"] * 2 * n, 200 + 1
        )

        # self.alpha = 1
        # print('Alpha:', self.alpha)

    def forward(self, rnn_output: torch.Tensor, data: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        indices, labels, fracs, other_indices = data

        reshaped_rnn_output = rnn_output.view(-1, self.config["size"])

        bias_tensor = torch.ones(
            reshaped_rnn_output.shape[0], 1, device=reshaped_rnn_output.device
        )

        rnn_with_bias = torch.cat([bias_tensor, reshaped_rnn_output], dim=1)

        subset_rnn_with_bias = rnn_with_bias[:, :201].contiguous()

        logits = embedding_dot.embedding_dot(
            rnn_with_bias, self.main_weights.weight, other_indices
        ) + embedding_dot.embedding_dot(
            subset_rnn_with_bias, self.sub_weights.weight, indices
        )

        frac_logits = logits[: len(fracs)]
        frac_labels = labels[: len(fracs)]

        frac_probs = torch.sigmoid(frac_logits) * fracs
        frac_loss = torch.nn.functional.binary_cross_entropy(
            frac_probs, frac_labels, reduction="sum"
        )

        nonfrac_logits = logits[len(fracs) :]
        nonfrac_labels = labels[len(fracs) :]

        nonfrac_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            nonfrac_logits, nonfrac_labels, reduction="sum"
        )

        total_loss = (frac_loss + nonfrac_loss) / len(labels)

        return logits, total_loss

    @classmethod
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        initial: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        a, b, c, d = initial
        a = torch.tensor(a, device=device, dtype=torch.int64)
        b = torch.tensor(b, device=device, dtype=torch.float)
        c = torch.tensor(c, device=device, dtype=torch.float)
        d = torch.tensor(d, device=device, dtype=torch.int64)
        return (a, b, c, d)


# a = SequentialTask({'size': 100, 'num_valid_targets': 100}, {})

# blah = torch.rand((19, 200, 100))

# t1 = torch.rand((19, 200, 100)) * 10 * 365
# t2 = torch.rand((19, 200, 100)) > 0.5
# t3 = torch.rand((19, 200, 100)) > 0.5

# b = a(blah, (t1, t2, t3))
