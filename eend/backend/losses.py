#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini, Lukas Burget, Mireia Diez)
# Copyright 2022 AUDIAS Universidad Autonoma de Madrid (author: Alicia Lozano-Diez)
# Licensed under the MIT license.

from itertools import permutations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.functional import logsigmoid
from scipy.optimize import linear_sum_assignment


def pit_loss_multispk(
        logits: List[torch.Tensor], target: List[torch.Tensor],
        n_speakers: np.ndarray, detach_attractor_loss: bool):
    if detach_attractor_loss:
        # -1's for speakers that do not have valid attractor
        for i in range(target.shape[0]):
            target[i, :, n_speakers[i]:] = -1 * torch.ones(
                          target.shape[1], target.shape[2]-n_speakers[i])

    logits_t = logits.detach().transpose(1, 2)
    cost_mxs = -logsigmoid(logits_t).bmm(target) - logsigmoid(-logits_t).bmm(1-target)

    max_n_speakers = max(n_speakers)

    for i, cost_mx in enumerate(cost_mxs.cpu().numpy()):
        if max_n_speakers > n_speakers[i]:
            max_value = np.absolute(cost_mx).sum()
            cost_mx[-(max_n_speakers-n_speakers[i]):] = max_value
            cost_mx[:, -(max_n_speakers-n_speakers[i]):] = max_value
        pred_alig, ref_alig = linear_sum_assignment(cost_mx)
        assert (np.all(pred_alig == np.arange(logits.shape[-1])))
        target[i, :] = target[i, :, ref_alig]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
             logits, target, reduction='none')

    loss[torch.where(target == -1)] = 0
    # normalize by sequence length
    loss = torch.sum(loss, axis=1) / (target != -1).sum(axis=1)
    for i in range(target.shape[0]):
        loss[i, n_speakers[i]:] = torch.zeros(loss.shape[1]-n_speakers[i])

    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss


def vad_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    # Take from reference ts only the speakers that do not correspond to -1
    # (-1 are padded frames), if the sum of their values is >0 there is speech
    vad_ts = (torch.sum((ts != -1)*ts, 2, keepdim=True) > 0).float()
    # We work on the probability space, not logits. We use silence probabilities
    ys_silence_probs = 1-torch.sigmoid(ys)
    # The probability of silence in the frame is the product of the
    # probability that each speaker is silent
    silence_prob = torch.prod(ys_silence_probs, 2, keepdim=True)
    # Estimate the loss. size=[batch_size, num_frames, 1]
    loss = F.binary_cross_entropy(silence_prob, 1-vad_ts, reduction='none')
    # "torch.max(ts, 2, keepdim=True)[0]" keeps the maximum along speaker dim
    # Invalid frames in the sequence (padding) will be -1, replace those
    # invalid positions by 0 so that those losses do not count
    loss[torch.where(torch.max(ts, 2, keepdim=True)[0] < 0)] = 0
    # normalize by sequence length
    # "torch.sum(loss, axis=1)" gives a value per batch
    # if torch.mean(ts,axis=2)==-1 then all speakers were invalid in the frame,
    # therefore we should not account for it
    # ts is size [batch_size, num_frames, num_spks]
    loss = torch.sum(loss, axis=1) / (torch.mean(ts, axis=2) != -1).sum(axis=1, keepdims=True)
    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss


def osd_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    aux = torch.clone(ts)
    aux[aux < 0] = 0  # nullify -1 positions
    nooverlap_ts = (torch.sum(aux, 2, keepdim=True) < 2).float()

    p_sil = 1 - torch.sigmoid(ys)

    # Imagine the case of 3 speakers for one specific frame, probability of no overlap will be
    # p_sil1 * p_sil2 * p_sil3 +
    # p_sp1  * p_sil2 * p_sil3 +
    # p_sil1 * p_sp2  * p_sil3 +
    # p_sil1 * p_sil2 * p_sp3
    # Then, to construct such matrix, we use prob_mask and inv_prob_mask which will look like (respectively)
    # 0 0 0        1 1 1
    # 1 0 0        0 1 1
    # 0 1 0        1 0 1
    # 0 0 1        1 1 0
    # Then, prob_mask * (1 - p_sil)[0,0] + inv_prob_mask * p_sil[0,0] will give us the following
    # (note that the first two dimensions are batchsize and sequence_length and the unsqueeze operators are there because of this)
    # p_sil1 p_sil2 p_sil3
    # p_sp1  p_sil2 p_sil3
    # p_sil1 p_sp2  p_sil3
    # p_sil1 p_sil2 p_sp3
    # So we only need to multiply along the last dimension and then sum along the second to last dimension.
    # Then, we will obtain p_nooverlap of dimensions (batchsize, sequence_length)
    mask = torch.cat((torch.zeros(1, ys.shape[2], device=ys.device), torch.eye(ys.shape[2], device=ys.device)), dim=0)
    prob_mask = mask.unsqueeze(0).unsqueeze(0).expand(ys.shape[0], ys.shape[1], -1, -1)
    inv_prob_mask = 1 - prob_mask
    p_nooverlap = torch.sum(torch.prod(prob_mask * (1 - p_sil).unsqueeze(2) + inv_prob_mask * p_sil.unsqueeze(2), dim=3), dim=2).unsqueeze(2)
    loss = F.binary_cross_entropy(p_nooverlap, nooverlap_ts, reduction='none')  # estimate the loss. size=[batch_size, num_framess, 1]

    loss[torch.where(torch.max(ts, 2, keepdim=True)[0] < 0)] = 0  # replace invalid positions by 0 so that those losses do not count
    # normalize by sequence length (not counting padding positions)
    loss = torch.sum(loss, axis=1) / (torch.mean(ts, axis=2) != -1).sum(axis=1, keepdims=True)  # torch.sum(loss, axis=1) gives a value per batch. if torch.mean(ts,axis=2)==-1 then all speakers were invalid in the frame, therefore we should not account for it. ts is size [batch_size, num_frames, num_spks]
    # normalize in batch so that all sequences count the same
    loss = torch.mean(loss)
    return loss


def batch_pit_n_speaker_loss(
    device: torch.device,
    ys: torch.Tensor,
    ts: torch.Tensor,
    n_speakers_list: List[int]
) -> Tuple[float, torch.Tensor]:
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = max(n_speakers_list)

    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = torch.stack([torch.roll(t, -shift, dims=1) for t in ts])
        # loss: (B, T, C)
        loss = F.binary_cross_entropy_with_logits(
            ys,
            ts_roll.float(),
            reduction='none').detach()
        # zero parts of sequences that correspond to padding
        loss[torch.where(ts_roll == -1)] = 0
        # sum over time: (B, C)
        loss = torch.sum(loss, axis=1)
        # normalize by sequence length
        loss = loss / (ts_roll != -1).sum(axis=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, axis=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t

    perms = np.asarray(list(permutations(range(max_n_speakers))), dtype="i")
    # y_ind: [0,1,2,3]
    y_ind = np.arange(max_n_speakers, dtype='i')
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = np.mod(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(torch.mean(losses[:, y_ind, t_ind], axis=1))
    # losses_perm: (B, Perm)
    losses_perm = torch.stack(losses_perm, axis=1)

    # masks: (B, Perms)
    def select_perm_indices(num: int, max_num: int) -> List[int]:
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [[x[:num] for x in perms].index(perm) for perm in sub_perms]

    masks = torch.full_like(losses_perm, float("Inf"))
    for i, _ in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    # normalize across batch
    min_loss = torch.mean(torch.min(losses_perm, dim=1)[0])

    min_indices = torch.argmin(losses_perm.detach(), axis=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(
        labels_perm, n_speakers_list)]

    return min_loss, labels_perm
