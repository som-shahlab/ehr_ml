import itertools
import math

import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

import copy


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(
            self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k))
        )
        nn.init.normal_(
            self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k))
        )
        nn.init.normal_(
            self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v))
        )

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5)
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = (
            q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        )  # (n*b) x lq x dk
        k = (
            k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        )  # (n*b) x lk x dk
        v = (
            v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        )  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.gelu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s, embed_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8),
        diagonal=1,
    )
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(
        sz_b, -1, -1
    )  # b x ls x ls

    return subsequent_mask


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, dec_input, slf_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
        self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1
    ):

        super().__init__()

        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    d_model, d_inner, n_head, d_k, d_v, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, tgt_seq):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask = slf_attn_mask_subseq.gt(0)

        # -- Forward
        dec_output = tgt_seq

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, slf_attn_mask=slf_attn_mask
            )

        return dec_output


class PatientRNN(nn.Module):
    def __init__(self, config, info):
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

        self.drop = nn.Dropout(config["dropout"])
        self.recurrent = self.config["encoder_type"] in ["gru", "lstm"]

        if self.recurrent:
            input_size = config["size"]
            model_class = (
                torch.nn.GRU
                if config["encoder_type"] == "gru"
                else torch.nn.LSTM
            )
            self.model = model_class(
                input_size=input_size,
                hidden_size=config["size"],
                num_layers=config["rnn_layers"],
                dropout=config["dropout"] if config["rnn_layers"] > 1 else 0,
            )
        else:
            self.model = Decoder(
                n_layers=6,
                n_head=8,
                d_k=64,
                d_v=64,
                d_model=config["size"],
                d_inner=2048,
                dropout=config["dropout"],
            )

    def forward(self, rnn_input):
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
            if self.recurrent
            else (self.config["size"] - 5 - 200)
        )

        embedded_non_text_codes = self.drop(
            self.input_code_embedding(all_non_text_codes, all_non_text_offsets)
        )[:, :size_for_embedding]

        embedded_non_text_codes1 = F.pad(
            self.drop(
                self.input_code_embedding1(
                    all_non_text_codes1, all_non_text_offsets1
                )
            ),
            pad=(size_for_embedding - ((self.config["size"] // 4) + 1), 0),
            mode="constant",
            value=0,
        )

        items = [
            a
            for a in [
                embedded_non_text_codes + embedded_non_text_codes1,
                all_day_information,
                all_positional_encoding if not self.recurrent else None,
            ]
            if a is not None
        ]

        combined_with_day_information = torch.cat(items, dim=1,)

        codes_split_by_patient = [
            combined_with_day_information.narrow(0, offset, length)
            for offset, length in all_lengths
        ]

        packed_sequence = nn.utils.rnn.pack_sequence(codes_split_by_patient)

        if self.recurrent:
            output, _ = self.model(packed_sequence)

            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )

            padded_output = self.drop(padded_output)

            return padded_output.contiguous()
        else:
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_sequence, batch_first=True
            )

            padded_output = padded_output.contiguous()

            return self.model(padded_output)

    @classmethod
    def finalize_data(cls, initial, device):
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
