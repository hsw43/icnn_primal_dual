#%%
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import simple_ICNNPrior
from utils import power, power_LL, proj_Lrelu, get_operators
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Problem Setup
img_size = 512
fwd_op, fbp_op, adjoint_op = get_operators()

# Load data
phantom = torch.load('data/phantom.npy').to(device)
sino = torch.load('data/sino.npy').to(device)
fbp = torch.load('data/fbp.npy').to(device)

# Load regularizer
regularizer = simple_ICNNPrior(1, 32, 5, img_size, False, device)
ckp = torch.load('weights/reg_icnn_ct.pt')
regularizer.load_state_dict(ckp)
#%%
x_shape = fbp.shape
z_shape = regularizer.icnn.wx(fbp).shape

gamma = 1 / 10
r = 5 / gamma

#Data Fidelity (KL)
def kl(x, y, r):
    return torch.sum(x - y + r) + torch.sum(y * torch.log(y / (x + r)))

#Def Scaled forward operator
def A(x_in):
    return fwd_op(x_in) / gamma

def AT(y_in):
    return adjoint_op(y_in) / gamma


def MA(u_in):
    return [A(u_in[0])]

def M0(u_in):
    return [regularizer.icnn.W0(u_in[0]), u_in[1]]

def M1(u_in):
    return [regularizer.icnn.WA(u_in[1])]

def MAT(v_in):
    return [AT(v_in[0]), torch.zeros(z_shape).to(device)]

def M0T(v_in):
    return [regularizer.icnn.W0T(v_in[0]), v_in[1]]

def M1T(v_in):
    return [torch.zeros(x_shape).to(device), regularizer.icnn.WAT(v_in[0])]

u_power = [torch.rand(x_shape).to(device), torch.rand(z_shape).to(device)]
M0norm = power_LL(M0, M0T, u_power)
M1norm = power_LL(M1, M1T, u_power)
MAnorm = power_LL(MA, MAT, u_power)
W0norm = power(regularizer.icnn.W0, regularizer.icnn.W0T, torch.rand(x_shape).to(device))
WAnorm = power(regularizer.icnn.WA, regularizer.icnn.WAT, torch.rand(z_shape).to(device))
opnorm = power(A, AT, torch.rand(x_shape).to(device))

# Scaled operators
def MA_new(u_in):
    return [A(u_in[0]) / MAnorm]

def M0_new(u_in):
    return [regularizer.icnn.W0(u_in[0]) / M0norm, u_in[1] / M0norm]

def M1_new(u_in):
    return [regularizer.icnn.WA(u_in[1]) / M1norm]

def MT_x(v_in):
    return AT(v_in[0][0]) / MAnorm + regularizer.icnn.W0T(v_in[1][0]) / M0norm

def MT_z(v_in):
    return v_in[1][1] / M0norm + regularizer.icnn.WAT(v_in[2][0]) / M1norm

#%%
#PD Test
reg_param = 600
max_iter = 500
b0 = regularizer.icnn.wx.bias.data.view(1, regularizer.icnn.n_filters, 1, 1)
bf1 = regularizer.icnn.fc1.bias.data
a_weight = reg_param * regularizer.icnn.fc2.weight.data
cmin = 0 * a_weight
cmax = a_weight * M1norm

loss_pd_all = []
psnr_pd_all_gt = []
x_recon_pd_500 = []
x_recon_pd_50 = []
data_pd_all = []
reg_pd_all = []
ca, c1, c2 = 500, 100, .1

with torch.no_grad():
    sigmaa = ca / (opnorm / MAnorm) ** 2
    sigma0 = c1 / (W0norm / M0norm) ** 2
    sigma1 = c2 / (WAnorm / M1norm) ** 2
    taux = 1 / (ca + c1)
    tauz = 1 / (sigma0 / W0norm ** 2 + c2)

    x = fbp.detach().clone().to(device)
    z = F.leaky_relu(regularizer.icnn.wx(x), 0.2).to(device)
    va = [torch.zeros_like(sino).to(device)]
    va = [1 - sino / (A(x) + r)]
    v1 = [torch.zeros_like(z), torch.zeros_like(z)]
    v_init = reg_param * regularizer.icnn.fc2.weight.data.clone()
    v_init[regularizer.icnn.WA(z) + bf1 <= 0] = 0
    v2 = [v_init * M1norm]
    v = [va, v1, v2]
    xold = x.clone()
    zold = z.clone()
    ubar = [2 * x - xold, 2 * z - zold]
    data_fid = kl(A(x), sino, r)
    reg = reg_param * regularizer.g(x)
    loss = data_fid + reg
    loss_pd_all = [loss.item()]
    data_pd_all = [data_fid.item()]
    reg_pd_all = [reg.item()]
    psnr_pd_all_gt = [compare_psnr(phantom.detach().cpu().numpy(), x.detach().cpu().numpy())]


    for i in range(max_iter):    
        vina = (v[0][0] + sigmaa * MA_new(ubar)[0])
        v[0][0] = (vina + MAnorm + sigmaa * r / MAnorm - torch.sqrt((vina - MAnorm + sigmaa * r / MAnorm)**2 + 4 * sigmaa * sino)) / 2
        vin1 = [a + sigma0 * b for a, b in zip(v[1], M0_new(ubar))]
        projv1 = proj_Lrelu([M0norm * vin1[0] / sigma0 + b0, M0norm * vin1[1] / sigma0], 0.2)
        v[1][0] = vin1[0] - sigma0 * (projv1[0] - b0) / M0norm
        v[1][1] = vin1[1] - sigma0 * projv1[1] / M0norm
        v[2][0] = torch.clamp((M1norm * (v[2][0] + sigma1 * M1_new(ubar)[0]) + sigma1 * bf1) / M1norm, cmin, cmax)
        x = torch.clamp(x - taux * MT_x(v), 0)
        z = z - tauz * MT_z(v)
        ubar = [2 * x - xold, 2 * z - zold]
        xold = x.clone()
        zold = z.clone()

        reg = reg_param * regularizer.g(x)
        data_fid = kl(A(x), sino, r)
        loss = data_fid + reg
        loss_pd_all.append(loss.item())
        data_pd_all.append(data_fid.item())
        reg_pd_all.append(reg.item())
        psnr_pd_all_gt.append(compare_psnr(phantom.cpu().numpy(), x.cpu().numpy()))
        recon_log = '[iter: {:d}/{:d}, var_loss: {:.10f}, fid: {:.10f}, reg: {:.10f}]'\
            .format(i+1, max_iter, loss.item(), data_fid.item(), reg.item())
        print(recon_log, end='\r')
        if i==50:
            x_recon_pd_50.append(x.detach().cpu())

    x_recon_pd_500.append(x.detach())
#%%
#GDC
lr_l = [0.0001, 0.0002, 0.0004, 0.0005]
psnr_gdc_all_gt = []
loss_gdc_all = []
data_gdc_all = []
reg_gdc_all = []
recon_gdc_all_500 = []
recon_gdc_all_50 = []

for lr in lr_l:
    print(f'\nGDC: lr={lr}')
    x_init = fbp.detach().clone()
    x_cvx = x_init.detach().clone().requires_grad_(True).to(device) 

    loss_all = []
    data_all = []
    reg_all = []
    psnr = compare_psnr(phantom.detach().cpu().numpy(),x_cvx.detach().cpu().numpy())
    psnr_all= [psnr.item()]

    
    for iter in np.arange(max_iter):
        data_loss = kl(A(x_cvx), sino, r)
        prior = reg_param * regularizer.g(x_cvx)
        variational_loss = data_loss + prior
        grad_fid = AT(1 - sino / (A(x_cvx) + r))
        grad_reg = torch.autograd.grad(prior, x_cvx)[0]
        x_cvx = x_cvx - lr * (grad_fid + grad_reg)
        with torch.no_grad():
            x_cvx.data.clamp_(0)
        
        loss_all.append(variational_loss.item())
        data_all.append(data_loss.item())
        reg_all.append(prior.item())
        psnr = compare_psnr(phantom.detach().cpu().numpy(),x_cvx.detach().cpu().numpy())
        psnr_all.append(psnr.item())
        recon_log = '[iter: {:d}/{:d}, var_loss: {:.6f}, data: {:.6f}, reg: {:.6f}]'\
        .format(iter+1, max_iter, variational_loss.item(), data_loss.item(), prior.item())

        print(recon_log, end='\r')
        if iter==50:
            recon_gdc_all_50.append(x_cvx.detach().clone().cpu())


    data_loss = kl(A(x_cvx), sino, r)
    prior = reg_param * regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all.append(variational_loss.item())
    data_all.append(data_loss.item())
    reg_all.append(prior.item())

    loss_gdc_all.append(loss_all)
    data_gdc_all.append(data_all)
    reg_gdc_all.append(reg_all)
    psnr_gdc_all_gt.append(psnr_all)
    recon_gdc_all_500.append(x_cvx.detach().clone().cpu())
#%%
#GDD
lr_l2 = [0.001, 0.003, 0.005, 0.007]
psnr_gdd_all_gt = []
loss_gdd_all = []
data_gdd_all = []
reg_gdd_all = []
recon_gdd_all_500 = []
recon_gdd_all_50 = []

for lr in lr_l2:
    print(f'\nGDD: Initial lr={lr}')
    x_init = fbp.detach().clone()
    x_cvx = x_init.detach().clone().requires_grad_(True).to(device)
    data_loss = kl(A(x_cvx), sino, r)
    prior = reg_param * regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all = []
    data_all = []
    reg_all = []
    psnr = compare_psnr(phantom.detach().cpu().numpy(), x_cvx.detach().cpu().numpy())
    psnr_all = [psnr.item()]

    for iter in np.arange(max_iter):
        alpha = lr/(1+iter)
        data_loss = kl(A(x_cvx), sino, r)
        prior = reg_param * regularizer.g(x_cvx)
        variational_loss = data_loss + prior
        grad_fid = AT(1 - sino / (A(x_cvx) + r))
        grad_reg = torch.autograd.grad(prior, x_cvx)[0]
        x_cvx = x_cvx - alpha * (grad_fid + grad_reg)
        with torch.no_grad():
            x_cvx.data.clamp_(0)

        loss_all.append(variational_loss.item())
        data_all.append(data_loss.item())
        reg_all.append(prior.item())
        psnr = compare_psnr(phantom.detach().cpu().numpy(), x_cvx.detach().cpu().numpy())
        psnr_all.append(psnr.item())
        recon_log = '[iter: {:d}/{:d}, var_loss: {:.6f}, data: {:.6f}, reg: {:.6f}]'\
        .format(iter+1, max_iter, variational_loss.item(), data_loss.item(),prior.item())

        print(recon_log, end='\r')
        if iter==50:
            recon_gdd_all_50.append(x_cvx.detach().clone().cpu())

    data_loss = kl(A(x_cvx), sino, r)
    prior = reg_param * regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all.append(variational_loss.item())
    data_all.append(data_loss.item())
    reg_all.append(prior.item())
    
    loss_gdd_all.append(loss_all)
    data_gdd_all.append(data_all)
    reg_gdd_all.append(reg_all)
    psnr_gdd_all_gt.append(psnr_all)
    recon_gdd_all_500.append(x_cvx.detach().clone().cpu())

#%%
# Visualizations
colors_l = [[0.37,0.77,0.79],[0.76,0.48,0.82],[0.93,0.69,0.13],[0.47,0.67,0.19],[0,0.45,0.74],[0.8500,0.33,0.1]]
#Comp plots (GDC)
xx = np.arange(501)
plt.semilogx(xx+1, loss_pd_all, color=colors_l[1], label='Proposed')
for j in range(len(loss_gdc_all)):
    plt.semilogx(xx+1, loss_gdc_all[j], label='Step-size:{:.5f}'.format(lr_l[j]))
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Objective value', fontsize=14)
plt.ylim([50000, 150000])
plt.grid()
plt.show()

#Comp plots (GDD)
plt.semilogx(xx+1, loss_pd_all, color=colors_l[1], label='Proposed')
for j in range(len(loss_gdd_all)):
    plt.semilogx(xx+1, loss_gdd_all[j], label='Initial step-size:{:.3f}'.format(lr_l2[j]))
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Objective value', fontsize=14)
plt.ylim([50000, 150000])
plt.grid()
plt.show()
# %%
#PSNR plots (GDC)
plt.semilogx(xx+1, psnr_pd_all_gt, color=colors_l[1], label='Proposed')
for j in range(len(psnr_gdc_all_gt)):
    plt.semilogx(xx+1, psnr_gdc_all_gt[j], label='Step-size:{:.5f}'.format(lr_l[j]))
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('PSNR', fontsize=14)
plt.ylim([21, 32])
plt.grid()
plt.show()

#PSNR plots (GDD)
plt.semilogx(xx+1, psnr_pd_all_gt, color=colors_l[1], label='Proposed')
for j in range(len(psnr_gdd_all_gt)):
    plt.semilogx(xx+1, psnr_gdd_all_gt[j], label='Initial step-size:{:.3f}'.format(lr_l2[j]))
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('PSNR', fontsize=14)
plt.ylim([21, 32])
plt.grid()
plt.show()
# %%
#recon
plt.imshow(x_recon_pd_50[0][0][0].clamp(0,1).detach().cpu(),cmap='bone')
plt.axis('off')
plt.show()
print('PD50:',compare_psnr(phantom.detach().cpu().numpy(),x_recon_pd_50[0].detach().cpu().numpy()))

plt.imshow(x_recon_pd_500[0][0][0].clamp(0,1).detach().cpu(),cmap='bone')
plt.axis('off')
plt.show()
print('PD500:',compare_psnr(phantom.detach().cpu().numpy(),x_recon_pd_500[0].detach().cpu().numpy()))

plt.imshow(recon_gdc_all_50[2][0][0].clamp(0,1).detach().cpu(),cmap='bone')
plt.axis('off')
plt.show()
print('GDC50:',compare_psnr(phantom.detach().cpu().numpy(),recon_gdc_all_50[2].detach().cpu().numpy()))

plt.imshow(recon_gdc_all_500[2][0][0].clamp(0,1).detach().cpu(),cmap='bone')
plt.axis('off')
plt.show()
print('GDC500:',compare_psnr(phantom.detach().cpu().numpy(),recon_gdc_all_500[2].detach().cpu().numpy()))

plt.imshow(recon_gdd_all_50[2][0][0].clamp(0,1).detach().cpu(),cmap='bone')
plt.axis('off')
plt.show()
print('GDD50:',compare_psnr(phantom.detach().cpu().numpy(),recon_gdd_all_50[2].detach().cpu().numpy()))

plt.imshow(recon_gdd_all_500[2][0][0].clamp(0,1).detach().cpu(),cmap='bone')
plt.axis('off')
plt.show()
print('GDD500:',compare_psnr(phantom.detach().cpu().numpy(),recon_gdd_all_500[2].detach().cpu().numpy()))
# %%
#Data plots (GDC)
plt.semilogx(xx+1, data_pd_all, color=colors_l[1], label='Proposed')
for j in range(len(loss_gdc_all)):
    plt.semilogx(xx+1, data_gdc_all[j], label='Step-size:{:.5f}'.format(lr_l[j]))
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Data fidelity', fontsize=14)
plt.ylim([20000, 250000])
plt.grid()
plt.show()

#Data plots (GDD)
plt.semilogx(xx+1, data_pd_all, color=colors_l[1], label='Proposed')
for j in range(len(loss_gdd_all)):
    plt.semilogx(xx+1, data_gdd_all[j], label='Initial step-size:{:.3f}'.format(lr_l2[j]))
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Data fidelity', fontsize=14)
plt.ylim([20000, 250000])
plt.grid()
plt.show()


#Reg plots (GDC))
plt.semilogx(xx+1, reg_pd_all, color=colors_l[1], label='Proposed')
for j in range(len(loss_gdc_all)-1):
    plt.semilogx(xx+1, reg_gdc_all[j], label='Step-size:{:.5f}'.format(lr_l[j]))
plt.semilogx(xx+1, reg_gdc_all[-1], label='Step-size:{:.5f}'.format(lr_l[-1]), alpha=0.9)
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Regularization', fontsize=14)
plt.ylim([16000, 62000])
plt.grid()
plt.show()

#Reg plots (GDD)
plt.semilogx(xx+1, reg_pd_all, color=colors_l[1], label='Proposed')
for j in range(len(loss_gdd_all)):
    plt.semilogx(xx+1, reg_gdd_all[j], label='Initial step-size:{:.3f}'.format(lr_l2[j]))
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Regularization', fontsize=14)
plt.ylim([16000, 62000])
plt.grid()
plt.show()
# %%
