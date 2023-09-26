#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2023 Brno University of Technology (author: Jiangyu Han, Federico Landini)
# Licensed under the MIT license.

"""
The reference of SAPEncoder:
https://github.com/funcwj/conv-tasnet/blob/master/nnet/conv_tas_net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import List, Tuple

from backend.losses import (
    pit_loss_multispk,
    osd_loss,
    vad_loss,
)
from common_utils.pad_utils import pad_labels

class Conv2dEncoder(nn.Module):
    def __init__(self, idim, odim):
        super(Conv2dEncoder, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, odim, (3, 7), (1, 5), (1, 0)),
            nn.ReLU(),
            nn.Conv2d(odim, odim, (3, 7), (1, 5), (1, 0)),
            nn.ReLU(),
        )
        self.down_dim = ((idim - 2)//5 - 2) // 5
        self.out = nn.Linear(odim * self.down_dim, odim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        """
        x = x.unsqueeze(1)  
        x = self.conv(x)
        b, c, t, n = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * n))
        return x  
    
class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, idim, time).
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # B x N x T => B x T x N
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # B x N x T => B x T x N
        x = torch.transpose(x, 1, 2)
        return x

class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = ChannelWiseLayerNorm(conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = ChannelWiseLayerNorm(conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x

class SAPEncoder(nn.Module):
    def __init__(self, odim=256, hidden=512):
        super(SAPEncoder, self).__init__()
        self.conv = nn.Conv1d(1, odim, 1) 
        self.conv1d_block = Conv1DBlock(odim, hidden)
        
    def forward(self, x):
        """
        input: B, T, 1
        out: B, T, N
        """
        x = torch.einsum('ijk->ikj', x)
        x = F.relu(self.conv(x)) 
        x = self.conv1d_block(x)     
        return torch.einsum('ijk->ikj', x)  

class MultiHeadSelfAttention(nn.Module):
    """ Multi head self-attention layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.linearQ = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearK = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearV = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearO = torch.nn.Linear(n_units, n_units, device=self.device)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) \
            / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = F.dropout(self.att, self.dropout)
        x = torch.matmul(p_att, v.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)

class PositionwiseFeedForward(nn.Module):
    """ Positionwise feed-forward layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        d_units: int,
        dropout: float
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(n_units, d_units, device=self.device)
        self.linear2 = torch.nn.Linear(d_units, n_units, device=self.device)
        self.dropout = dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.dropout(F.relu(self.linear1(x)), self.dropout))

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        device: torch.device,
        idim: int,
        n_layers: int,
        n_units: int,
        e_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.linear_in = torch.nn.Linear(idim, n_units, device=self.device)
        self.lnorm_in = torch.nn.LayerNorm(n_units, device=self.device)
        self.n_layers = n_layers
        self.dropout = dropout
        for i in range(n_layers):
            setattr(
                self,
                '{}{:d}'.format("lnorm1_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("self_att_", i),
                MultiHeadSelfAttention(self.device, n_units, h, dropout)
            )
            setattr(
                self,
                '{}{:d}'.format("lnorm2_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("ff_", i),
                PositionwiseFeedForward(self.device, n_units, e_units, dropout)
            )
        self.lnorm_out = torch.nn.LayerNorm(n_units, device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + F.dropout(s, self.dropout)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + F.dropout(s, self.dropout)
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)

class DiaCorrect(nn.Module):
    def __init__(
        self,
        device: torch.device,
        in_size: int = 345,
        n_units: int = 256,
        e_units: int = 2048,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        vad_loss_weight: float = 0.0,
        osd_loss_weight: float = 0.0,
        n_speakers: int = 2
    ) -> None:
        """ Self-attention-based DiaCorrect model.
        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          e_units (int): Number of units in FeedForwardLayer
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          vad_loss_weight (float) : weight for vad_loss
          osd_loss_weight (float) : weight for osd_loss
          n_speakers (int): Number of speakers, default 2
        """
        self.device = device
        super(DiaCorrect, self).__init__()
        self.speech_enc = Conv2dEncoder(in_size, n_units).to(self.device)
        self.sap_enc = SAPEncoder(n_units).to(self.device)
        self.transformer_dec = TransformerEncoder(
            self.device, n_units*3, n_layers, n_units, e_units, n_heads, dropout
        )
        self.linear_dec = nn.Linear(n_units, n_speakers).to(self.device)
        self.vad_loss_weight = vad_loss_weight
        self.osd_loss_weight = osd_loss_weight

    def forward(
        self,
        xs: torch.Tensor,
        sap: torch.Tensor,
        sigmoid: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_len, _ = xs.shape
        speech_emb = self.speech_enc(xs)
        spk_num = sap.shape[-1]
        sap_emb = torch.cat([self.sap_enc(torch.unsqueeze(sap[:, :, c], -1)) 
                                                for c in range(spk_num)], -1)
        ys = self.transformer_dec(torch.cat([speech_emb, sap_emb], -1))
        ys = ys.reshape(batch_size, time_len, -1)
        ys = self.linear_dec(ys)
        if sigmoid:
            ys = [torch.sigmoid(y) for y in ys]
        return ys

    def get_loss(
        self,
        ys: torch.Tensor,
        target: torch.Tensor,
        n_speakers: List[int],
        vad_loss_weight: float,
        osd_loss_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_n_speakers = max(n_speakers)
        ts_padded = pad_labels(target, max_n_speakers)
        ts_padded = torch.stack(ts_padded)
        ys_padded = pad_labels(ys, max_n_speakers)
        ys_padded = torch.stack(ys_padded)

        loss = pit_loss_multispk(
            ys_padded, ts_padded, n_speakers, detach_attractor_loss=False)
        vad_loss_value = vad_loss(ys, target)
        osd_loss_value = osd_loss(ys, target)

        return loss + vad_loss_value * vad_loss_weight + \
            osd_loss_value * osd_loss_weight, loss


if __name__ == '__main__':
    model = DiaCorrect(torch.device('cpu'))
    xs = torch.randn(1, 500, 345)
    sap = torch.randn(1, 500, 2)
    ts = torch.randn(1, 500, 2)
    ys = model(xs, ts, sap)
    print(ys[0].shape)
        
