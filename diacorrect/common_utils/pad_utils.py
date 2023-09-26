#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import torch
from typing import Tuple, List

def pad_labels(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts_padded.append(torch.cat((t, -1 * torch.ones((
                t.shape[0], out_size - t.shape[1]))), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            ts_padded.append(t[:, :out_size].float())
        else:
            ts_padded.append(t.float())
    return ts_padded

def pad_sequence(
    features: List[torch.Tensor],
    labels: List[torch.Tensor],
    sap: List[torch.Tensor],
    seq_len: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features_padded = []
    labels_padded = []
    sap_padded = []
    assert len(features) == len(labels) == len(sap), (
        f"Features and labels in batch were expected to match but got "
        "{len(features)} features and {len(labels)} labels.")
    for i, _ in enumerate(features):
        assert features[i].shape[0] == labels[i].shape[0] == sap[i].shape[0], (
            f"Length of features and labels sap were expected to match but got "
            "{features[i].shape[0]} and {labels[i].shape[0]} and {sap[i].shape[0]}")
        length = features[i].shape[0]
        if length < seq_len:
            extend = seq_len - length
            features_padded.append(torch.cat((features[i], -torch.ones((
                extend, features[i].shape[1]))), dim=0))
            labels_padded.append(torch.cat((labels[i], -torch.ones((
                extend, labels[i].shape[1]))), dim=0))
            sap_padded.append(torch.cat((sap[i], -torch.ones((
                extend, sap[i].shape[1]))), dim=0))
        elif length > seq_len:
            raise (f"Sequence of length {length} was received but only "
                   "{seq_len} was expected.")
        else:
            features_padded.append(features[i])
            labels_padded.append(labels[i])
            sap_padded.append(sap[i])
    return features_padded, labels_padded, sap_padded