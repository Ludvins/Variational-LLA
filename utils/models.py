
import torch
from torch import nn
import math

import torch.nn.functional as F


def get_mlp(input_dim, output_dim, inner_dims, activation, dropout = False,
            final_layer_activation = None, device = None, dtype = None):
    torch.manual_seed(2147483647)
    
    layers = []
    dims = [input_dim] + inner_dims + [output_dim]
    for i, (_in, _out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(
            torch.nn.Linear(_in, _out, device = device, dtype = dtype)
        )

        if i != len(dims) -2:       
            layers.append(activation()) 
            if dropout:
                layers.append(
                    torch.nn.Dropout(0.1)
                )
            
    if final_layer_activation is not None:
        layers.append(final_layer_activation())
    
    model = torch.nn.Sequential(*layers)
    print(model)
    setattr(model, 'output_size', output_dim)
    setattr(model, "implements_jacobian", False)
    return model

class LinearLayerJacobian(nn.Module):
    def __init__(self, in_features, out_features, device, dtype):
        super(LinearLayerJacobian, self).__init__()

        self.in_features = in_features
        self.n_outputs = out_features
        self.device = device
        self.dtype = dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.empty((in_features, out_features), **factory_kwargs)
        self.bias = torch.empty(1, out_features, **factory_kwargs)
        self.activation = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = x @ self.weight + self.bias
        return out, x

    def backward(self, cache, dout, pi_dout = None, inducing =False):
        """
        Computes the accumulated Jacobian of the layer and returns the Jacobian wrt
        the parameters at this layer.

        Arguments
        ---------
        dout : torch tensor of shape (batch_size, output_dimension_model, output_dimension_layer)
               Contains the Jacobian of the output the full model wrt the output
               of this layer.
        cache : tuple of (torch tensor of shape (batch_size, input_features), torch tensor of shape (input_features, output_features)) 
                Contains the input values that went trough the forward pass of
                this layer wrt which dout is computed and the weights used in that forward pass.
        outputs: torch tensor of shape (batch_size, o)
                 Contains the outputs wrt which the Jacobian needs to be computed. Related to 
                 output_dimension_model in the first argument.

        Returns
        -------
        dout : torch tensor of shape (batch_size, dim)
               Contains the JAcobian of the output of the full model wrt the input of
               this layer.
        dparams : torch tensor of shape (batch_size, 0, input_features * output_features + output_features)
                  Contains the Jacobian wrt the parameters of this layer.

        """

        if pi_dout is None:
            return dout @ self.weight.T

        return dout @ self.weight.T, pi_dout @ self.weight.T

class TanhJacobian(nn.Module):
    def __init__(self):
        super(TanhJacobian, self).__init__()
        self.activation = True

    def forward(self, x):
        """
        Computes the tanh of the given input.

        Arguments
        ---------
        x : torch.tensor of shape (batch_size, dim)

        Returns
        -------
        output : torch tensor of shape (batch_size, dim)
                 Contains the application of tanh to x
        cache : torch tensor of shape (batch_size, dim)
                Contains the given input.
        """
        cache = x
        return torch.tanh(x), cache

    def backward(self, cache, dout, pi_dout = None, inducing = False):
        """
        Computes the acuumulated Jacobian of the layer. Let f denote the output of a full
        model using this layer, a denote the output of this layer and o denote its input.

        This function computes:
            df/do = df/da * da/do

        Arguments
        ---------
        dout : torch tensor of shape (batch_size, dim)
               Contains the Jacobian of the output the full model wrt the output
               of this layer.
        cache : torch tensor of shape (batch_size, dim)
                Contains the input values that went trough the forward pass of
                this layer wrt which dout is computed.

        Returns
        -------
        dout : torch tensor of shape (batch_size, dim)
               Contains the JAcobian of the output of the full model wrt the input of
               this layer.

        """
        
        
        x = cache
        derivative = 1 - torch.tanh(x)**2

        if inducing is True:

            dout = torch.einsum("...ap, ...ap -> ...ap", dout, derivative)
            return dout

        dout = torch.einsum("a...p, ...ap -> a...p", dout, derivative)
        if pi_dout is not None:
            pi_dout = torch.einsum("a...p, ap -> a...p", pi_dout, derivative)
            return dout, pi_dout
    
        return dout

def create_ad_hoc_mlp(mlp):

    linear_layers = []
    for layer in mlp:
        if isinstance(layer, nn.Linear):
            linear_layers.append(layer)
    
    mlp_ad_hoc = MLP(mlp[0].weight.shape[1], mlp[-1].weight.shape[0], len(linear_layers)-1,
                     mlp[0].weight.shape[0], 
                     device =  mlp[-1].weight.device, 
                     dtype =  mlp[-1].weight.dtype)
    for i in range(0, len(linear_layers)):
        mlp_ad_hoc.layers[2*i].weight = linear_layers[i].weight.T
        mlp_ad_hoc.layers[2*i].bias = linear_layers[i].bias.unsqueeze(0)

    return mlp_ad_hoc

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, n_units, 
                 device, dtype):
        """ The MLP must have the first and last layers as FC.
        :param n_inputs: input dim
        :param n_outputs: output dim
        :param n_layers: layer num = n_layers + 2
        :param n_units: the dimension of hidden layers
        :param nonlinear: nonlinear function
        """
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.output_size = n_outputs
        self.n_layers = n_layers
        self.n_units = n_units
        self.dtype = dtype
        self.implements_jacobian = True
        self.device = device
        
        # create layers
        layers = []
        for i in range(n_layers):
            if i == 0:
                units = n_inputs
            else:
                units = n_units
                
            layers.append(LinearLayerJacobian(units, n_units, device = device, dtype = dtype))
            layers.append(TanhJacobian())
        layers.append(LinearLayerJacobian(n_units, n_outputs, device = device, dtype = dtype))
        
        
        self.layers = nn.Sequential(*layers)
        self.implements_jacobian = True
        
    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        
        return x
    
    def get_kernel(self, x, z):
     

        
        # Arrays to store data from forward pass
        x_caches = []
        z_caches = []
        
        # Forward pass
        for layer in self.layers:
            x, cache = layer(x)
            x_caches.append(cache)
            z, cache = layer(z)
            z_caches.append(cache)
            
            
        # Initial derivatives of each input, indentity matrices
        # Shapes (points, output_dim, output_dim)
        dx = torch.tile(torch.diag(torch.ones(x.shape[1], dtype = self.dtype, device = self.device)).unsqueeze(0),
                        (x.shape[0], 1, 1))
        dz = torch.tile(torch.diag(torch.ones(z.shape[1], dtype = self.dtype, device = self.device)).unsqueeze(0),
                        (z.shape[0], 1, 1))
        
        # Stores the kernel matrix on x and z
        Kxz = torch.zeros(x.shape[0], z.shape[0],
                          x.shape[1], z.shape[1], device = self.device, dtype = self.dtype)

        # Backward pass
        for i in range(len(self.layers)-1, -1, -1):
            
            # Get stored inputs for the layer
            i_x = x_caches[i] 
            i_z = z_caches[i] 
            
            # If layer is not activation layer 
            if self.layers[i].activation == False:
                
                # Pre-compute the product of the inputs
                #  a and b are used for point dimensions and i for units dimension
                i_xz_product = torch.einsum("ai, bi-> ab", i_x, i_z)
                   
                
                ###################################################
                ####################### Kxz #######################
                
                # Pre-compute outputs product
                # a and b are point dimension and o is output sumsampled dimensions and p is unit dimensions
                output_product = torch.einsum("aop, bup -> abou", dx, dz)
                
                # Weight contribution
                Kxz += output_product * i_xz_product.unsqueeze(-1).unsqueeze(-1)
                # Bias contribution
                Kxz += output_product
                
            # Backward pass return the gradeint wrt the input (for the recursive calls)
            #  and the gradients wrt the parameters of that layer.
            # dx shape (batch_size, outputs_x, n_units)
            # pi_dx shape (batch_size, n_units)
            dx = self.layers[i].backward(i_x, dx)
            # dz shape (n_inducing, n_units)
            dz = self.layers[i].backward(i_z, dz)
                
        return Kxz
    
    def get_jacobian(self, x):
             
        # Arrays to store data from forward pass
        x_caches = []
        
        # Forward pass
        for layer in self.layers:
            x, cache = layer(x)
            x_caches.append(cache)
            
        
        # Initial derivatives of each input, indentity matrices
        # Shapes (points, output_dim, output_dim)
        dx = torch.tile(torch.diag(torch.ones(x.shape[1], dtype = self.dtype, device = self.device)).unsqueeze(0),
                        (x.shape[0], 1, 1))

        # Stores the kernel matrix on x and z
        J = []

        # Backward pass
        for i in range(len(self.layers)-1, -1, -1):
            
            # Get stored inputs for the layer
            i_x = x_caches[i] 
            
            # If layer is not activation layer 
            if self.layers[i].activation == False:
            
                w = i_x.unsqueeze(1).unsqueeze(-1) * dx.unsqueeze(-2)

                J.append(w.flatten(-2, -1))
                
                J.append(dx)
                
            # Backward pass return the gradeint wrt the input (for the recursive calls)
            #  and the gradients wrt the parameters of that layer.
            # dx shape (batch_size, outputs_x, n_units)
            # pi_dx shape (batch_size, n_units)
            dx = self.layers[i].backward(i_x, dx)

        return torch.cat(J, -1)
    
    def get_jacobian_on_outputs(self, x, outputs_x):
        # Arrays to store data from forward pass
        x_caches = []
        
        # Forward pass
        for layer in self.layers:
            x, cache = layer(x)
            x_caches.append(cache)
            
        
        # Initial derivatives of each input, indentity matrices
        # Shapes (points, output_dim, output_dim)
        dx = torch.tile(torch.diag(torch.ones(x.shape[1], dtype = self.dtype, device = self.device)).unsqueeze(0),
                        (x.shape[0], 1, 1))
        dx = torch.gather(dx, 1, torch.tile(outputs_x.unsqueeze(-1).unsqueeze(-1), (1,1,dx.shape[-1]))).squeeze(1)

        # Stores the kernel matrix on x and z
        J = []

        # Backward pass
        for i in range(len(self.layers)-1, -1, -1):
            
            # Get stored inputs for the layer
            i_x = x_caches[i] 
            
            # If layer is not activation layer 
            if self.layers[i].activation == False:
                
                w = i_x.unsqueeze(-1) * dx.unsqueeze(1)

                J.append(w.flatten(-2, -1))
                
                J.append(dx)
                
            # Backward pass return the gradeint wrt the input (for the recursive calls)
            #  and the gradients wrt the parameters of that layer.
            # dx shape (batch_size, outputs_x, n_units)
            # pi_dx shape (batch_size, n_units)
            dx = self.layers[i].backward(i_x, dx)

        return torch.cat(J, -1)
    

    def get_full_kernels(self, x, z, outputs_z):
        # x shape (batch_size, input_dim)
        # z shape (num_inducing, input_dim)
        # outputs_x shape (batch_size, num_outputs_x (usually 2))
        # outputs_z shape (num_inducing)



        # Arrays to store data from forward pass
        x_caches = []
        z_caches = []

        # Forward pass
        for layer in self.layers:
            x, cache = layer(x)
            x_caches.append(cache)
            z, cache = layer(z)
            z_caches.append(cache)


        # Initial derivatives of each input, indentity matrices
        # Shapes (points, output_dim, output_dim)
        dx = torch.tile(torch.diag(torch.ones(x.shape[1], dtype = self.dtype, device = self.device)).unsqueeze(0),
                        (x.shape[0], 1, 1))
        dz = torch.tile(torch.diag(torch.ones(z.shape[1], dtype = self.dtype, device = self.device)).unsqueeze(0),
                        (z.shape[0], 1, 1))

        # Initialize Kernel Matrices for inputs and inducing locations
        # Stores the diagonal of the kernel matrix on the input x
        Kxx = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device = self.device, dtype = self.dtype)
        # Stores the kernel matrix on x and z
        Kxz = torch.zeros(x.shape[0], z.shape[0],
                          x.shape[1], device = self.device, dtype = self.dtype)
        # Stores the kernel matrix on the inducing locations z
        Kzz = torch.zeros(z.shape[0], z.shape[0], device = self.device, dtype = self.dtype)

        # Reduce the second dimension to consider only the desired outputs for each input
        # Shape (inducing_points, output_dim)
        dz = torch.gather(dz, 1, torch.tile(outputs_z.unsqueeze(-1).unsqueeze(-1), (1,1,dz.shape[-1]))).squeeze(1)

        # Backward pass
        for i in range(len(self.layers)-1, -1, -1):

            # Get stored inputs for the layer
            i_x = x_caches[i]
            i_z = z_caches[i]

            # If layer is not activation layer
            if self.layers[i].activation == False:

                # Pre-compute the product of the inputs
                #  a and b are used for point dimensions and i for units dimension
                i_x_product = torch.einsum("ai, ai-> a", i_x, i_x)
                i_xz_product = torch.einsum("ai, bi-> ab", i_x, i_z)
                i_z_product = torch.einsum("ai, bi-> ab", i_z, i_z)


                ###################################################
                #####################  Kzz ########################

                # Pre-compute outputs product
                # a and b are inducing dimension and o is units dimension
                output_product = torch.einsum("ao, bo -> ab", dz, dz)

                # Weight contribution
                Kzz += output_product * i_z_product
                # Bias contribution
                Kzz += output_product

                ###################################################
                ##################### diag(Kxx) ###################

                # Pre-compute outputs product
                # a is point dimension and o is output sumsampled dimensions and p is unit dimensions
                output_product = torch.einsum("aop, aup -> aou", dx, dx)

                # Weight contribution
                Kxx += output_product * i_x_product.unsqueeze(-1).unsqueeze(-1)
                # Bias contribution
                Kxx += output_product

                ###################################################
                ####################### Kxz #######################

                # Pre-compute outputs product
                # a and b are point dimension and o is output sumsampled dimensions and p is unit dimensions
                output_product = torch.einsum("aop, bp -> abo", dx, dz)

                # Weight contribution
                Kxz += output_product * i_xz_product.unsqueeze(-1)
                # Bias contribution
                Kxz += output_product

            # Backward pass return the gradeint wrt the input (for the recursive calls)
            #  and the gradients wrt the parameters of that layer.
            # dx shape (batch_size, outputs_x, n_units)
            # pi_dx shape (batch_size, n_units)
            dx = self.layers[i].backward(i_x, dx)
            # dz shape (n_inducing, n_units)
            dz = self.layers[i].backward(i_z, dz)

        return x, Kxx, Kxz, Kzz
    

'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016,
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional


cifar10_pretrained_weight_urls = {
    'resnet20': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
    'resnet32': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
    'resnet44': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
    'resnet56': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

cifar100_pretrained_weight_urls = {
    'resnet20': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
    'resnet32': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
    'resnet44': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
    'resnet56': 'http://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(
    arch: str,
    layers: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = True,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    
    return model

def get_resnet(model, num_classes):
    
    if num_classes != 10 and num_classes != 100:
        raise ValueError("Number of classes must be 10 or 100.")
    
    model_urls = cifar10_pretrained_weight_urls if num_classes == 10 else cifar100_pretrained_weight_urls
    layers = {
        "resnet20": [3]*3,
        "resnet32": [5]*3,
        "resnet44": [7]*3,
        "resnet56": [9]*3,
    }
    model = _resnet(model, layers[model], model_urls, num_classes = num_classes)
    setattr(model, 'output_size', num_classes)
    model.eval()
    return model
