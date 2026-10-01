'''
Copyright 2024 Elijah Bolluyt.

Implementation of Pseudo-Invertible Neural Networks (Psi-NN)

Usage:

PsiNNConv2d:
    A pseudo-invertible 2D convolutional layer, as described in the original paper.  The forw() and back() functions perform convolution and transpose convolution (analogous to torch.nn.Conv2d and torch.nn.ConvTranspose2d).  For compatibility with other PyTorch classes, the forward() function points to one of the two directional operations specified by the "direction" initialization argument.

    Most of the arguments directly correspond to their counterparts in torch.nn.Conv2d; the "direction" argument is added to determine the operation performed by the forward() function.

AE_Classifier:
    An autoencoder with classification head, for use in semi-supervised classification problems. The model is built using bidirectional PsiNN layers, which the encoder and decoder share.  Use the C() method to perform classification with the encoder layers, and the AE() method to reconstruct a sample through the full autoencoder.
AE_Baseline_Classifier:
    An autoencoder with classification head, for use in semi-supervised classification problems. The model is built with convolutional layers, with the encoder and decoder separately parameterized.  This is provided for reference and comparison to the AE_Classifier model.  Use the C() method to perform classification with the encoder layers, and the AE() method to reconstruct a sample through the full autoencoder.
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d, ConvTranspose2d

import math

def to_tuple(a):
    if isinstance(a,tuple):
        return a
    else:
        return (a,a)
        
def rightInverse(A):
    AAt = torch.matmul(A,torch.transpose(A,0,1))
    AAt1 = torch.inverse(AAt)
    return torch.matmul(torch.transpose(A,0,1),AAt1)
        
class PsiNNConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=1,bias=True,direction=1):
        super(PsiNNConv2d,self).__init__()
        ''' direction = 1 -> Conv2d; direction = -1 -> ConvTranspose2d '''
        self.direction = direction
        ''' Different values of dilation and groups are not yet supported '''
        dilation = 1
        groups = 1
        self.output_padding = to_tuple(output_padding)
        self.kernel_size = to_tuple(kernel_size)
        self.stride = to_tuple(stride)
        self.padding = to_tuple(padding)
        self.dilation = to_tuple(dilation)
        
        self.in_channels = in_channels if direction == 1 else out_channels
        self.out_channels = out_channels if direction == 1 else in_channels
        
        if self.in_channels*self.kernel_size[0]*self.kernel_size[1] > self.out_channels:
            forward_direction = -1
        else:
            forward_direction = 1
        self.forward_direction = forward_direction
        self.unfold_forward = nn.Unfold(self.kernel_size,self.dilation,self.padding,self.stride)
        self.unfold_backward = nn.Unfold((1,1),1,0,1)
        if forward_direction == 1:
            self.w = nn.Parameter(torch.Tensor(self.in_channels*self.kernel_size[0]*self.kernel_size[1],self.out_channels))
        else:
            self.w = nn.Parameter(torch.Tensor(self.out_channels,self.in_channels*self.kernel_size[0]*self.kernel_size[1]))
        
        bound = math.sqrt(groups/(self.in_channels*self.kernel_size[0]*self.kernel_size[1]))
        torch.nn.init.uniform_(self.w,-bound,bound)

        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(1,self.out_channels,1,1))
            torch.nn.init.uniform_(self.b,-bound,bound)
        else:
            self.b = 0
    def forw(self,x):
        H_out = (x.shape[2] + 2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0] + 1
        W_out = (x.shape[3] + 2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1] + 1
        H_out,W_out = int(H_out),int(W_out)
        spatial_size = (x.shape[2],x.shape[3])
        L = H_out*W_out
        x = self.unfold_forward(x)
        x = x.transpose(1,2).reshape(-1,self.in_channels*self.kernel_size[0]*self.kernel_size[1])
        
        if self.forward_direction == 1:
            w = self.w
        else:
            w = rightInverse(self.w)
        
        x = torch.mm(x,w)
        x = x.reshape(-1,L,self.out_channels)
        x = x.transpose(1,2)
        x = F.fold(x,output_size=(H_out,W_out),kernel_size=1,dilation=1,padding=0,stride=1)
        x = x + self.b
        return x
    def back(self,x):
        x = x - self.b
        H_out = (x.shape[2]-1)*self.stride[0] - 2*self.padding[0]+self.kernel_size[0]+self.output_padding[0]
        W_out = (x.shape[3]-1)*self.stride[1] - 2*self.padding[1]+self.kernel_size[1]+self.output_padding[1]
        x = self.unfold_backward(x)
        L = x.shape[2]
        
        if self.forward_direction == -1:
            w = self.w
        else:
            w = rightInverse(self.w)
        x = x.transpose(1,2).reshape(-1,self.out_channels)
        x = torch.mm(x,w)
        x = x.reshape(-1,L,self.in_channels*self.kernel_size[0]*self.kernel_size[1])
        x = x.transpose(1,2)
        
        x = F.fold(x,output_size=(H_out,W_out),kernel_size=self.kernel_size,dilation=1,padding=self.padding,stride=self.stride)
        
        return x
    def forward(self,x):
        if self.direction == -1:
            return self.back(x)
        elif self.direction == 1:
            return self.forw(x)

class AE_Classifier(nn.Module):
	def __init__(self,n_channels,n_classes, nf=16, k=3, use_dropout=False):
		super(AE_Classifier,self).__init__()
		self.C5 = PsiNNConv2d(nf*8,n_classes,2,1,0,bias=False,direction=1)
		self.C4 = PsiNNConv2d(nf*8,nf*4,k,2,1,1,bias=False,direction=-1)
		self.C3 = PsiNNConv2d(nf*4,nf*2,k,2,1,1,bias=False,direction=-1)
		self.C2 = PsiNNConv2d(nf*2,nf*1,k,2,1,1,bias=False,direction=-1)
		self.C1 = PsiNNConv2d(nf*1,n_channels,k,2,1,1,bias=False,direction=-1)
		
		self.BN = nn.ModuleList([nn.BatchNorm2d(nf*i) for i in [1,2,4,8]])
		self.AE_BN = nn.ModuleList([nn.BatchNorm2d(nf*i) for i in [4,2,1]])
		
		if use_dropout:
			self.drop = nn.Dropout(0.3)
		else:
			self.drop = nn.Identity()
		
	def C(self,x):
		x = self.C1.forw(x)
		x = self.BN[0](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C2.forw(x)
		x = self.BN[1](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C3.forw(x)
		x = self.BN[2](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C4.forw(x)
		x = self.BN[3](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C5.forw(x)
		
		return x.squeeze()
		
	def AE(self,x):
		# forward through classifier, except final layer
		x = self.C1.forw(x)
		x = self.BN[0](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C2.forw(x)
		x = self.BN[1](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C3.forw(x)
		x = self.BN[2](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C4.forw(x)
		x = self.BN[3](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		# backward through classifier
		x = self.C4.back(x)
		x = self.AE_BN[0](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C3.back(x)
		x = self.AE_BN[1](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C2.back(x)
		x = self.AE_BN[2](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C1.back(x)
		
		return x
	def forward(self,x):
		return self.C(x)
		
class AE_Baseline_Classifier(nn.Module):
	def __init__(self,n_channels,n_classes, nf=16, k=3, use_dropout=False):
		super(AE_Baseline_Classifier,self).__init__()
		self.C5 = Conv2d(nf*8,n_classes,2,1,0,bias=False)
		self.C4 = Conv2d(nf*4,nf*8,k,2,1,1,bias=False)
		self.C3 = Conv2d(nf*2,nf*4,k,2,1,1,bias=False)
		self.C2 = Conv2d(nf*1,nf*2,k,2,1,1,bias=False)
		self.C1 = Conv2d(n_channels,nf*1,k,2,1,1,bias=False)
		
		#self.C5i = ConvTranspose2d(n_classes,nf*8,2,1,0,bias=False)
		self.C4i = ConvTranspose2d(nf*8,nf*4,k,2,1,1,bias=False)
		self.C3i = ConvTranspose2d(nf*4,nf*2,k,2,1,1,bias=False)
		self.C2i = ConvTranspose2d(nf*2,nf*1,k,2,1,1,bias=False)
		self.C1i = ConvTranspose2d(nf*1,n_channels,k,2,1,1,bias=False)
		
		self.BN = nn.ModuleList([nn.BatchNorm2d(nf*i) for i in [1,2,4,8]])
		self.AE_BN = nn.ModuleList([nn.BatchNorm2d(nf*i) for i in [4,2,1]])
		if use_dropout:
			self.drop = nn.Dropout(0.3)
		else:
			self.drop = nn.Identity()
		
	def C(self,x):
		x = self.C1(x)
		x = self.BN[0](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C2(x)
		x = self.BN[1](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C3(x)
		x = self.BN[2](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C4(x)
		x = self.BN[3](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C5(x)
		
		return x.squeeze()
	
	def AE(self,x):
		x = self.C1(x)
		x = self.BN[0](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C2(x)
		x = self.BN[1](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C3(x)
		x = self.BN[2](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C4(x)
		x = self.BN[3](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		# backward
		x = self.C4i(x)
		x = self.AE_BN[0](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C3i(x)
		x = self.AE_BN[1](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C2i(x)
		x = self.AE_BN[2](x)
		x = F.leaky_relu(x,0.2)
		x = self.drop(x)
		
		x = self.C1i(x)
		
		return x
	def forward(self,x):
		return self.C(x)
