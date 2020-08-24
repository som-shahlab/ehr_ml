import itertools
import math
from typing import Any, List, Mapping, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import transformer
from torch.nn.parameter import Parameter


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 10,
        num_encoder_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        print("Got", d_model, nhead, num_encoder_layers)

        encoder_layer = transformer.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = transformer.LayerNorm(d_model)
        self.encoder = transformer.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # type: ignore
        device = src.device
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        return self.encoder(src, mask)


class PatientRNN(nn.Module):
    def __init__(self, config: Mapping[str, Any], info: Mapping[str, Any]):
        super().__init__()
        self.config = config
        self.info = info

        self.input_code_embedding = nn.EmbeddingBag(
            config["num_first"] + 1, config["size"] + 1, mode="mean"
        )

        self.input_code_embedding1 = nn.EmbeddingBag(
            config["num_second"] + 1, (config["size"] // 4) + 1, mode="mean"
        )

        self.input_code_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.input_code_embedding1.weight.data.normal_(mean=0.0, std=0.02)

        self.norm = nn.LayerNorm(config["size"])
        self.drop = nn.Dropout(config["dropout"])

        self.model: nn.Module

        if config["use_gru"]:
            input_size = config["size"]
            self.model = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=config["size"],
                num_layers=config["gru_layers"],
                dropout=config["dropout"] if config["gru_layers"] > 1 else 0,
            )

        else:
            print("Transformer")
            self.model = Encoder(d_model=config["size"])

    def forward(self, rnn_input: Sequence[Any]) -> torch.Tensor:  # type: ignore
        (
            all_non_text_codes,
            all_non_text_offsets,
            all_non_text_codes1,
            all_non_text_offsets1,
            all_day_information,
            all_positional_encoding,
            all_lengths,
        ) = rnn_input

        size_for_embedding = (
            (self.config["size"] - 5)
            if self.config["use_gru"]
            else (self.config["size"] - 5 - 200)
        )

        embedded_non_text_codes = self.input_code_embedding(
            all_non_text_codes, all_non_text_offsets
        )[:, :size_for_embedding]

        embedded_non_text_codes1 = F.pad(
            self.input_code_embedding1(
                all_non_text_codes1, all_non_text_offsets1
            ),
            pad=[size_for_embedding - ((self.config["size"] // 4) + 1), 0],
            mode="constant",
            value=0,
        )

        items = [
            a
            for a in [
                embedded_non_text_codes + embedded_non_text_codes1,
                all_day_information,
                all_positional_encoding if not self.config["use_gru"] else None,
            ]
            if a is not None
        ]

        combined_with_day_information = torch.cat(items, dim=1,)

        # print(all_day_information.shape)
        # print(all_positional_encoding.shape)
        # print(all_non_text_codes.shape)
        # print(all_non_text_offsets.shape)

        # print(combined_with_day_information.shape)
        # print(all_lengths)

        codes_split_by_patient = [
            combined_with_day_information.narrow(0, offset, length)
            for offset, length in all_lengths
        ]

        packed_sequence = nn.utils.rnn.pack_sequence(codes_split_by_patient)

        if self.config["use_gru"]:
            output, _ = self.model(packed_sequence)

            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )

            padded_output = self.drop(padded_output)

            return padded_output.contiguous()
        else:
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_sequence, batch_first=False
            )

            padded_output = padded_output.contiguous()

            result = self.model(self.norm(padded_output))

            result = result.permute(1, 0, 2).contiguous()

            return result

    @classmethod
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        initial: Sequence[Any],
    ) -> Sequence[Any]:
        (
            all_non_text_codes,
            all_non_text_offsets,
            all_non_text_codes1,
            all_non_text_offsets1,
            all_day_information,
            all_positional_encoding,
            all_lengths,
        ) = initial

        all_non_text_codes = torch.tensor(
            all_non_text_codes, dtype=torch.long, device=device
        )
        all_non_text_offsets = torch.tensor(
            all_non_text_offsets, dtype=torch.long, device=device
        )
        all_non_text_codes1 = torch.tensor(
            all_non_text_codes1, dtype=torch.long, device=device
        )
        all_non_text_offsets1 = torch.tensor(
            all_non_text_offsets1, dtype=torch.long, device=device
        )

        all_day_information = torch.tensor(
            all_day_information, dtype=torch.float, device=device
        )

        all_positional_encoding = torch.tensor(
            all_positional_encoding, dtype=torch.float, device=device
        )

        all_lengths = [(int(a), int(b)) for (a, b) in all_lengths]

        return (
            all_non_text_codes,
            all_non_text_offsets,
            all_non_text_codes1,
            all_non_text_offsets1,
            all_day_information,
            all_positional_encoding,
            all_lengths,
        )
