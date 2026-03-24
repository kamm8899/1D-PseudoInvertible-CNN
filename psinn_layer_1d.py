'''
Copyright 2026 Jessica Kamman/Jessica Sinn.

1D adaptation of Pseudo-Invertible Neural Networks (Psi-NN)

Usage:

PsiNNConv1d:
    A pseudo-invertible 1D convolutional layer. The forw() and back() functions
    perform convolution and transpose convolution (analogous to torch.nn.Conv1d
    and torch.nn.ConvTranspose1d). For compatibility with other PyTorch classes,
    the forward() function points to one of the two directional operations
    specified by the "direction" initialization argument.

    Arguments correspond to their torch.nn.Conv1d counterparts; the "direction"
    argument determines which operation forward() calls (1 = Conv1d, -1 = ConvTranspose1d).

AE_Classifier1d:
    An autoencoder with classification head for semi-supervised classification,
    built using bidirectional PsiNN layers whose weights are shared between encoder
    and decoder. Use C() for classification and AE() for reconstruction.

AE_Baseline_Classifier1d:
    An autoencoder with classification head built with separately parameterized
    Conv1d / ConvTranspose1d layers. Provided as a reference baseline.
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d

import math


def rightInverse(A):
    '''Right inverse: A^T (A A^T)^{-1}. Requires A to be wide (more cols than rows).'''
    AAt = torch.matmul(A, torch.transpose(A, 0, 1))
    AAt1 = torch.inverse(AAt)
    return torch.matmul(torch.transpose(A, 0, 1), AAt1)


class PsiNNConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=1, bias=True, direction=1):
        super(PsiNNConv1d, self).__init__()
        '''direction = 1 -> Conv1d; direction = -1 -> ConvTranspose1d'''
        self.direction = direction
        '''Different values of dilation and groups are not yet supported'''
        dilation = 1
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Swap in/out channels when operating in reverse direction,
        # so the stored weight always corresponds to the forward (Conv1d) orientation.
        self.in_channels = in_channels if direction == 1 else out_channels
        self.out_channels = out_channels if direction == 1 else in_channels

        # Choose which direction stores w directly vs. rightInverse(w).
        # The stored matrix must be wide (more columns than rows) for rightInverse to exist.
        #   in_ch*k > out_ch  ->  store w as (out_ch, in_ch*k)  [wide]
        #   in_ch*k <= out_ch ->  store w as (in_ch*k, out_ch)  [wide]
        if self.in_channels * self.kernel_size > self.out_channels:
            forward_direction = -1
        else:
            forward_direction = 1
        self.forward_direction = forward_direction

        if forward_direction == 1:
            self.w = nn.Parameter(torch.Tensor(self.in_channels * self.kernel_size, self.out_channels))
        else:
            self.w = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels * self.kernel_size))

        bound = math.sqrt(1.0 / (self.in_channels * self.kernel_size))
        torch.nn.init.uniform_(self.w, -bound, bound)

        self.has_bias = bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, self.out_channels, 1))
            torch.nn.init.uniform_(self.b, -bound, bound)
        else:
            self.b = 0

    def forw(self, x):
        '''Forward convolution: (N, in_channels, L) -> (N, out_channels, L_out)'''
        N, C, L = x.shape
        L_out = (L + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # Unfold input into overlapping patches.
        # F.pad then unfold: (N, in_channels, L_out, kernel_size)
        x = F.pad(x, (self.padding, self.padding))
        x = x.unfold(2, self.kernel_size, self.stride)

        # (N, L_out, in_channels, kernel_size) -> (N*L_out, in_channels*kernel_size)
        x = x.permute(0, 2, 1, 3).reshape(N * L_out, C * self.kernel_size)

        w = self.w if self.forward_direction == 1 else rightInverse(self.w)

        x = torch.mm(x, w)  # (N*L_out, out_channels)
        x = x.reshape(N, L_out, self.out_channels).transpose(1, 2)  # (N, out_channels, L_out)
        return x + self.b

    def back(self, x):
        '''Transpose convolution: (N, out_channels, L_out) -> (N, in_channels, L)'''
        x = x - self.b
        N, C, L_out = x.shape
        L_in = (L_out - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # (N, out_channels, L_out) -> (N*L_out, out_channels)
        x = x.transpose(1, 2).reshape(N * L_out, self.out_channels)

        w = self.w if self.forward_direction == -1 else rightInverse(self.w)

        x = torch.mm(x, w)  # (N*L_out, in_channels*kernel_size)
        x = x.reshape(N, L_out, self.in_channels * self.kernel_size).transpose(1, 2)
        # x: (N, in_channels*kernel_size, L_out)

        # Fold patches back into a sequence.
        # F.fold input shape: (N, C*kH*kW, L_blocks)
        # Use height=1 to map the 1D fold onto the 2D F.fold API:
        #   C=in_channels, kH=1, kW=kernel_size, L_blocks=L_out
        x = F.fold(
            x,
            output_size=(1, L_in),
            kernel_size=(1, self.kernel_size),
            dilation=(1, self.dilation),
            padding=(0, self.padding),
            stride=(1, self.stride),
        )  # (N, in_channels, 1, L_in)
        return x.squeeze(2)  # (N, in_channels, L_in)

    def forward(self, x):
        if self.direction == -1:
            return self.back(x)
        elif self.direction == 1:
            return self.forw(x)


class AE_Classifier1d(nn.Module):
    def __init__(self, n_channels, n_classes, nf=16, k=3, use_dropout=False):
        super(AE_Classifier1d, self).__init__()
        self.C5 = PsiNNConv1d(nf * 8, n_classes, 2, 1, 0, bias=False, direction=1)
        self.C4 = PsiNNConv1d(nf * 8, nf * 4, k, 2, 1, 1, bias=False, direction=-1)
        self.C3 = PsiNNConv1d(nf * 4, nf * 2, k, 2, 1, 1, bias=False, direction=-1)
        self.C2 = PsiNNConv1d(nf * 2, nf * 1, k, 2, 1, 1, bias=False, direction=-1)
        self.C1 = PsiNNConv1d(nf * 1, n_channels, k, 2, 1, 1, bias=False, direction=-1)

        self.BN = nn.ModuleList([nn.BatchNorm1d(nf * i) for i in [1, 2, 4, 8]])
        self.AE_BN = nn.ModuleList([nn.BatchNorm1d(nf * i) for i in [4, 2, 1]])

        self.drop = nn.Dropout(0.3) if use_dropout else nn.Identity()

    def C(self, x):
        x = self.C1.forw(x);  x = self.BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2.forw(x);  x = self.BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C3.forw(x);  x = self.BN[2](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C4.forw(x);  x = self.BN[3](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C5.forw(x)
        return x.squeeze()

    def AE(self, x):
        # Encode
        x = self.C1.forw(x);  x = self.BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2.forw(x);  x = self.BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C3.forw(x);  x = self.BN[2](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C4.forw(x);  x = self.BN[3](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        # Decode (shared weights, reverse direction)
        x = self.C4.back(x);  x = self.AE_BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C3.back(x);  x = self.AE_BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2.back(x);  x = self.AE_BN[2](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C1.back(x)
        return x

    def forward(self, x):
        return self.C(x)


class AE_Baseline_Classifier1d(nn.Module):
    def __init__(self, n_channels, n_classes, nf=16, k=3, use_dropout=False):
        super(AE_Baseline_Classifier1d, self).__init__()
        self.C5 = Conv1d(nf * 8, n_classes, 2, 1, 0, bias=False)
        self.C4 = Conv1d(nf * 4, nf * 8, k, 2, 1, bias=False)
        self.C3 = Conv1d(nf * 2, nf * 4, k, 2, 1, bias=False)
        self.C2 = Conv1d(nf * 1, nf * 2, k, 2, 1, bias=False)
        self.C1 = Conv1d(n_channels, nf * 1, k, 2, 1, bias=False)

        self.C4i = ConvTranspose1d(nf * 8, nf * 4, k, 2, 1, 1, bias=False)
        self.C3i = ConvTranspose1d(nf * 4, nf * 2, k, 2, 1, 1, bias=False)
        self.C2i = ConvTranspose1d(nf * 2, nf * 1, k, 2, 1, 1, bias=False)
        self.C1i = ConvTranspose1d(nf * 1, n_channels, k, 2, 1, 1, bias=False)

        self.BN = nn.ModuleList([nn.BatchNorm1d(nf * i) for i in [1, 2, 4, 8]])
        self.AE_BN = nn.ModuleList([nn.BatchNorm1d(nf * i) for i in [4, 2, 1]])

        self.drop = nn.Dropout(0.3) if use_dropout else nn.Identity()

    def C(self, x):
        x = self.C1(x);  x = self.BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2(x);  x = self.BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C3(x);  x = self.BN[2](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C4(x);  x = self.BN[3](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C5(x)
        return x.squeeze()

    def AE(self, x):
        # Encode
        x = self.C1(x);  x = self.BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2(x);  x = self.BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C3(x);  x = self.BN[2](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C4(x);  x = self.BN[3](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        # Decode
        x = self.C4i(x);  x = self.AE_BN[0](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C3i(x);  x = self.AE_BN[1](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C2i(x);  x = self.AE_BN[2](x);  x = F.leaky_relu(x, 0.2);  x = self.drop(x)
        x = self.C1i(x)
        return x

    def forward(self, x):
        return self.C(x)
