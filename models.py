import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.optim import Prior
import torch.nn.utils.parametrize as P

# Simple ICNN 
class simple_ICNN(nn.Module):
    def __init__(self,n_channels,n_filters,kernel_size,img_size,smoothed,device):
        super(simple_ICNN,self).__init__()

        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        #first convolutional layer
        self.wx = nn.Conv2d(self.n_channels, self.n_filters, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=True)
        #average pooling layer to reduce the feature map size
        self.avg_pool = nn.AvgPool2d(kernel_size=16, stride=16, padding=0)
        self.avg_ones = torch.ones(self.n_filters, 1, 16, 16, device=device) / 256.
        ####dense layers in the end############
        self.dim1 = int(img_size/16)**2*n_filters
        self.dim2 = img_size
        self.avg_size = int(img_size/16)
        self.fc1 = nn.Linear(in_features=self.dim1, out_features=self.dim2, bias=True)
        self.fc2 = nn.Linear(in_features=self.dim2, out_features=1, bias=True)
        ######################################
        self.smoothing = 0.01
        self.smoothed = smoothed
        self.negative_slope = 0.2 
        
        self.device = device
        
        if smoothed:
            self.act1 = (
                lambda x: torch.clip(
                    x, 0, self.smoothing
                )
                ** 2
                / (2 * self.smoothing)
                + torch.clip(x, self.smoothing)
                - self.smoothing
            ) # smoothed ReLU
            
            self.act2 = (
                lambda x: self.negative_slope * x + (1 - self.negative_slope) * self.act1(x)
            ) # smoothed LeakyReLU
        else:
            self.act1 = torch.nn.ReLU()
            self.act2 = torch.nn.LeakyReLU(self.negative_slope)
    
    def forward(self, x):
        self.zero_clip_weights()
        z = self.act2(self.wx(x))
        z_avg_pool = self.avg_pool(z)
        z_flat = z_avg_pool.view(z_avg_pool.size(0), -1)
        z1 = self.act1(self.fc1(z_flat))
        z1 = self.fc2(z1)
        return torch.mean(z1.view(z1.shape[0], -1), dim=1).view(z1.shape[0], -1)

    # linear operators for Primal-Dual algorithm
    def W0(self, x_in):
        return F.conv2d(x_in, self.wx.weight.data, padding=self.padding)

    def W0T(self, xd_in):
        return F.conv_transpose2d(xd_in, self.wx.weight.data, padding=self.padding)

    def WA(self, z_in):
        z_avg = self.avg_pool(z_in)
        return F.linear(z_avg.view(z_avg.shape[0], -1), self.fc1.weight.data)

    def WAT(self, zd_in):
        z_avg = F.linear(zd_in, self.fc1.weight.data.t()).view(zd_in.shape[0], self.n_filters, self.avg_size, self.avg_size)
        return F.conv_transpose2d(z_avg, self.avg_ones, stride=16, groups=self.n_filters)

    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self):     
        self.fc1.weight.data.clamp_(0)
        self.fc2.weight.data.clamp_(0)
        return self 

    #a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.01):
        self.fc1.weight.data = min_val + (max_val - min_val) * torch.rand(self.dim2, self.dim1).to(self.device)
        self.fc2.weight.data = min_val + (max_val - min_val) * torch.rand(1, self.dim2).to(self.device)
        return self


class deeper_ICNN(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size, img_size, n_layers, device):
        super(deeper_ICNN, self).__init__()

        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.n_layers = n_layers

        #first convolutional layer
        self.wx = nn.Conv2d(self.n_channels, self.n_filters, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=True)
        #these layers should have non-negative weights
        self.wz = nn.ModuleList([nn.Conv2d(self.n_filters, self.n_filters, kernel_size=self.kernel_size, stride=1, padding=2, bias=True)\
                                 for i in range(self.n_layers)])
        
        #average pooling layer to reduce the feature map size
        self.avg_pool = nn.AvgPool2d(kernel_size=16, stride=16, padding=0)
        self.avg_ones = torch.ones(self.n_filters, 1, 16, 16, device=device) / 256.
        ####dense layers in the end############
        self.dim1 = int(img_size/16)**2*n_filters
        self.dim2 = img_size
        self.avg_size = int(img_size/16)
        self.fc1 = nn.Linear(in_features=self.dim1, out_features=self.dim2, bias=True)
        self.fc2 = nn.Linear(in_features=self.dim2, out_features=1, bias=True)
        self.negative_slope = 0.2 
        
        self.device = device
        
        
    def forward(self, x):
        z = torch.nn.functional.leaky_relu(self.wx(x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z), negative_slope=self.negative_slope)
        z_avg_pool = self.avg_pool(z)
        z_flat = z_avg_pool.view(z_avg_pool.size(0),-1)
        
        z1 = torch.nn.functional.relu(self.fc1(z_flat))
        z1 = self.fc2(z1)
        
        return z1
    
    # linear operators for Primal-Dual algorithm
    def W0(self, x_in):
        return F.conv2d(x_in, self.wx.weight.data, padding=self.padding)

    def W0T(self, xd_in):
        return F.conv_transpose2d(xd_in, self.wx.weight.data, padding=self.padding)

    def W1(self, z_in):
        return F.conv2d(z_in, self.wz[0].weight.data, padding=2)
    
    def W1T(self, zd_in):
        return F.conv_transpose2d(zd_in, self.wz[0].weight.data, padding=2)

    def W2(self, z_in):
        return F.conv2d(z_in, self.wz[1].weight.data, padding=2)

    def W2T(self, zd_in):
        return F.conv_transpose2d(zd_in, self.wz[1].weight.data, padding=2)

    def WA(self, z_in):
        z_avg = self.avg_pool(z_in)
        return F.linear(z_avg.view(z_avg.shape[0], -1), self.fc1.weight.data)

    def WAT(self, zd_in):
        z_avg = F.linear(zd_in, self.fc1.weight.data.t()).view(zd_in.shape[0], self.n_filters, self.avg_size, self.avg_size)
        return F.conv_transpose2d(z_avg, self.avg_ones, stride=16, groups=self.n_filters)

      
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        
        self.fc1.weight.data.clamp_(0)
        self.fc2.weight.data.clamp_(0)
        return self 

    #a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val)\
            * torch.rand(self.n_filters, self.n_filters, self.kernel_size, self.kernel_size).to(self.device)

        self.fc1.weight.data = min_val + (max_val - min_val)*torch.rand(self.dim2, self.dim1).to(self.device)
        self.fc2.weight.data = min_val + (max_val - min_val)*torch.rand(1, self.dim2).to(self.device)
        return self


class simple_ICNNPrior(Prior):
    def __init__(self, n_channels, n_filters, kernel_size, img_size, smoothed, device):
        super().__init__()
        self.icnn = simple_ICNN(n_channels, n_filters, kernel_size, img_size, smoothed, device).to(device)
        self.icnn.initialize_weights()

    def g(self, x):
        return self.icnn(x)

    def grad(self, x, get_energy=False):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            z = torch.sum(self.g(x_))
            grad = torch.autograd.grad(z, x_, create_graph=True)[0]
        if get_energy:
            return z, grad
        return grad
    
    
class deeper_ICNNPrior(Prior):
    def __init__(self, n_channels, n_filters, kernel_size, img_size, n_layers, device):
        super().__init__()
        self.icnn = deeper_ICNN(n_channels, n_filters, kernel_size, img_size, n_layers, device).to(device)
        self.icnn.initialize_weights()

    def g(self, x):
        return self.icnn(x)

    def grad(self, x, get_energy=False):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            z = torch.sum(self.g(x_))
            grad = torch.autograd.grad(z, x_, create_graph=True)[0]
        if get_energy:
            return z, grad
        return grad