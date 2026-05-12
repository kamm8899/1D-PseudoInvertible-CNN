'''
Copyright 2026 Jessica Kamman/Jessica Sinn.

PsiNN architecture mirroring the Pablos et al. CAE design:
    - Input: (N, 1, 1024) — I channel only (matches CAE)
    - Encoder: 3 PsiNNConv1d layers, 1->16->64->128, kernel=5, stride=2
    - Decoder: same 3 layers reversed using pseudoinverse weights
    - Linear output — no sigmoid, matches normalized input range
    - Size mismatch corrected via F.interpolate

Differs from AE_Classifier1d in psinn_layer_1d.py:
    - Single channel input (not 2-channel IQ)
    - 3 encoder layers instead of 4
    - Channel progression matches CAE (1->16->64->128)
    - No classification head
'''

import torch
from torch import nn
from torch.nn import functional as F

from psinn_layer_1d import PsiNNConv1d


class AE_Pablos1d(nn.Module):
    '''
    PsiNN autoencoder matching the Pablos et al. CAE architecture.
    Input: (N, 1, 1024) - I channel only.
    Encoder: 3 PsiNNConv1d layers (1->16->64->128, k=5, s=2).
    Decoder: same 3 layers reversed using pseudoinverse weights.
    Output: linear (no sigmoid), size-corrected to match input length.
    '''
    def __init__(self, nf=16, k=5, use_dropout=False):
        super(AE_Pablos1d, self).__init__()

        # Layer definitions follow AE_Classifier1d convention:
        # PsiNNConv1d(out, in, k, ..., direction=-1)
        # forw() maps: in -> out (encode)
        # back() maps: out -> in (decode, pseudoinverse)
        self.C1 = PsiNNConv1d(nf,      1,      k, 2, 1, 1, bias=False, direction=-1)  # 1->16
        self.C2 = PsiNNConv1d(nf * 4,  nf,     k, 2, 1, 1, bias=False, direction=-1)  # 16->64
        self.C3 = PsiNNConv1d(nf * 8,  nf * 4, k, 2, 1, 1, bias=False, direction=-1)  # 64->128

        self.BN    = nn.ModuleList([nn.BatchNorm1d(nf), nn.BatchNorm1d(nf * 4), nn.BatchNorm1d(nf * 8)])
        self.AE_BN = nn.ModuleList([nn.BatchNorm1d(nf * 4), nn.BatchNorm1d(nf)])

        self.drop = nn.Dropout(0.3) if use_dropout else nn.Identity()

    def AE(self, x):
        L_in = x.shape[-1]

        # Encode
        x = self.C1.forw(x);  x = self.BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2.forw(x);  x = self.BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C3.forw(x);  x = self.BN[2](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)

        # Decode — pseudoinverse weights, reverse order, linear output
        x = self.C3.back(x);  x = self.AE_BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2.back(x);  x = self.AE_BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C1.back(x)

        if x.shape[-1] != L_in:
            x = F.interpolate(x, size=L_in, mode='linear', align_corners=False)
        return x

    def forward(self, x):
        return self.AE(x)
