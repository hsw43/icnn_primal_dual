##############################################################
########################Inpaint###############################
##############################################################
#%%
import torch
import torch.nn.functional as F
from models import simple_ICNNPrior
from utils import power
from deepinv.physics import Inpainting, GaussianNoise
from deepinv.optim import L2
import cv2
import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Problem Setup
img_size = 256
noise_level = 0.03
mask = torch.load('data/mask.npy').to(device)
physics = Inpainting(mask.shape[1:],mask,device=device,noise_model=GaussianNoise(sigma=noise_level))
data_fidelity = L2(sigma=1.0)
lmbd = 0.1

img = cv2.imread('data/inpaint.png')
img = skimage.color.rgb2gray(img)
img = torch.from_numpy(img).view(1,1,img_size,img_size).float().to(device)
torch.manual_seed(0)
y = (img+0.03*torch.randn_like(img))*mask

# ICNN Prior Setup
smoothed = True #True for smoothed ICNN, False for non-smoothed ICNN
smoothing = 0.01 if smoothed else 0.0

# Load regularizer
regularizer = simple_ICNNPrior(1,32,5,img_size,False,device)
ckp = torch.load('weights/reg_smoothed_icnn_ip_ar.pt')
regularizer.load_state_dict(ckp)
#%%
def cubic(a, b, c, d):
    f = ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0
    g = (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0
    h = ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)
    
    R = -(g / 2.0) + torch.sqrt(h)
    S = torch.sign(R) * torch.pow(torch.abs(R), 1 / 3.0)
    T = -(g / 2.0) - torch.sqrt(h)
    U = torch.sign(T) * torch.pow(torch.abs(T), 1 / 3.0)

    return (S + U) - (b / (3.0 * a))
    
# Projection to epigraph 
def proj_epi(x0, x1, reg):
    alpha = reg.icnn.negative_slope
    if reg.icnn.smoothed:
        omega = reg.icnn.smoothing
        lower = -x0 / alpha
        mid = -x0 + omega * (3 + alpha) / 2

        act2_x0 = reg.icnn.act2(x0)

        cond1 = (x1 < alpha * x0) & (x1 <= lower)
        cond2 = (act2_x0 > x1) & (x1 > lower) & (x1 < mid)
        cond3 = (act2_x0 > x1) & (x1 >= mid)

        o1_case1 = (x0 + alpha * x1) / (1 + alpha ** 2)
        #o2_case1 = (x1 + alpha * x0) / (1 + alpha ** 2)

        a = (1 - alpha) ** 2 / (2 * omega ** 2)
        b = 3 * alpha * (1 - alpha) / (2 * omega)
        c = 1 + alpha ** 2 - (1 - alpha) * x1 / omega
        d = -(x0 + alpha * x1)
        o1_case2 = cubic(a, b, c, d)
        o2_case2 = reg.icnn.act2(o1_case2)

        o1_case3 = (x0 + x1) / 2. + (1 - alpha) * omega / 4
        o2_case3 = (x1 + x0) / 2. - (1 - alpha) * omega / 4
        o1 = torch.where(cond1, o1_case1,
                torch.where(cond2, o1_case2,
                torch.where(cond3, o1_case3, x0)))
        o2 = torch.where(cond1, alpha*o1_case1,
                torch.where(cond2, o2_case2,
                torch.where(cond3, o2_case3, x1)))
    else:
        
        o_zero = torch.zeros_like(x0)
        # Condition masks
        cond1 = torch.abs(x1) < x0
        cond2 = (x1 < alpha * x0) & (x1 < -x0 / alpha)
        cond3 = (x1 >= -x0 / alpha) & (x1 <= -x0)
        # Precompute values
        o_temp1 = (x0 + x1) / 2.
        o_temp2 = (x0 + alpha * x1) / (1 + alpha ** 2)
        # Use torch.where for vectorized assignment
        o1 = torch.where(cond1, o_temp1,
                    torch.where(cond2, o_temp2,
                    torch.where(cond3, o_zero, x0)))
        o2 = torch.where(cond1, o_temp1,
                    torch.where(cond2, alpha*o_temp2,
                    torch.where(cond3, o_zero, x1)))    
        
    return o1, o2

##############################################################################################################
# PD
x = y.clone()
z = regularizer.icnn.act2(regularizer.icnn.wx(x)).detach()
b0 = regularizer.icnn.wx.bias.data.view(1, regularizer.icnn.n_filters, 1, 1)
bf1 = regularizer.icnn.fc1.bias.data
a_weight = lmbd * regularizer.icnn.fc2.weight.data
omega = regularizer.icnn.smoothing
cmin = 0*a_weight
cmax = a_weight
mask = physics.mask.data[0]
loss_pd_all = []
psnr_pd_all_gt = []
x_recon_pd_300 = []
x_recon_pd_60 = []
max_iter = 300

c1, c2 = 0.01, 0.001
x_power = torch.randn(1, 1, 256, 256, device=device)
z_power = torch.randn(1, 32, 256, 256, device=device)
W0norm = power(regularizer.icnn.W0, regularizer.icnn.W0T, x_power)
WAnorm = power(regularizer.icnn.WA, regularizer.icnn.WAT, z_power)

sigma0 = c1 / W0norm ** 2
sigma1 = c2 / WAnorm ** 2
taux = 1 / (sigma0 * W0norm ** 2)
tauz = 1 / (sigma0 + sigma1 * WAnorm ** 2)

with torch.no_grad():
    v11, v12 = torch.zeros_like(z), torch.zeros_like(z)
    v2 = lmbd * regularizer.icnn.fc2.weight.data.clone()
    #v2 = v2.expand(x.shape[0], -1)
    v2[regularizer.icnn.WA(z) + bf1 <= 0] = 0
    xold, zold = x.clone(), z.clone()
    xbar, zbar = x.clone(), z.clone()

    reg = lmbd * regularizer.g(x)
    data_fid = .5 * torch.norm(physics.A(x) - y) ** 2
    loss = data_fid + reg
    loss_pd_all.append(loss.item())
    psnr_pd_all_gt.append(compare_psnr(img.cpu().numpy(), x.cpu().numpy()))

    for iter in range(max_iter):

        v_in1, v_in2 = v11 + sigma0 * regularizer.icnn.W0(xbar), v12 + sigma0 * zbar
        projv11, projv12= proj_epi(v_in1 / sigma0 + b0, v_in2 / sigma0, regularizer)
        v11, v12 = v_in1 - sigma0 * (projv11 - b0), v_in2 - sigma0 * projv12
        v2 = torch.clamp((a_weight * (v2 + sigma1 * (regularizer.icnn.WA(zbar) + bf1))) / (sigma1 * omega + a_weight), cmin, cmax)

        x = (x - taux * regularizer.icnn.W0T(v11) + taux * physics.A_adjoint(y)) / (1 + taux * mask)
        z = z - tauz * (v12 + regularizer.icnn.WAT(v2))

        xbar, zbar = 2 * x - xold, 2 * z - zold
        xold, zold = x, z

        reg = lmbd * regularizer.g(x)
        data_fid = .5 * torch.norm(physics.A(x) - y) ** 2
        loss = data_fid + reg
        loss_pd_all.append(loss.item())
        grad = (data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)).detach()
        psnr_pd_all_gt.append(compare_psnr(img.cpu().numpy(), x.cpu().numpy()))
        recon_log = 'iter: {:d}/{:d},var_loss: {:.10f},grad: {:.4f},PSNR: {:.2f}'\
                .format(iter+1, max_iter, loss.item(), grad.norm().item(), psnr_pd_all_gt[-1])
        print(recon_log,end='\r')
        
        if iter==60:
            x_recon_pd_60.append(x.detach().clone().cpu())
x_recon_pd_300.append(x.detach().clone().cpu())

#%%
# nmAPG
from nmAPG import reconstruct_nmAPG

x_recon_nmapg_60, _ = reconstruct_nmAPG(y,img,physics,data_fidelity,regularizer,lmbd,1e-1,60,1e-10,progress=True,return_stats=True)
x_recon_nmapg_300, stats = reconstruct_nmAPG(y,img,physics,data_fidelity,regularizer,lmbd,1e-1,300,1e-10,progress=True,return_stats=True)
#%%
# GDC
lr_l = [1.5]
max_iter = 300
loss_gdc_all = []
x_recon_gdc_300 = []
x_recon_gdc_60 = []
psnr_gdc_all_gt = []
for lr in lr_l:
    print(f'\nGDC: lr={lr}')
    x_init = y.clamp(0,1).detach().clone().to(device)
    x_cvx = x_init.clone().detach().requires_grad_(True).to(device) 
    x_optimizer = torch.optim.SGD([x_cvx], lr=lr)

    data_loss = .5 * torch.norm(physics.A(x_cvx) - y) ** 2
    prior = lmbd * regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    
    loss_all = []
    psnr_all_gt = [compare_psnr(img.detach().cpu().numpy(), x_cvx.detach().cpu().numpy())]

    for iter in range(max_iter):
        x_optimizer.zero_grad()
        data_loss = .5 * torch.norm(physics.A(x_cvx) - y) ** 2
        prior = lmbd * regularizer.g(x_cvx)
        variational_loss = data_loss + prior
        grad = data_fidelity.grad(x_cvx, y, physics).detach() + lmbd * regularizer.grad(x_cvx).detach()
        x_cvx = x_cvx - lr * grad

        recon_log = 'iter: {:d}/{:d}, var_loss: {:.6f}, data: {:.6f}, reg: {:.6f}'\
        .format(iter+1, max_iter, variational_loss.item(), data_loss.item(), prior.item())
        print(recon_log, end='\r')
        loss_all.append(variational_loss.item())
        psnr_all_gt.append(compare_psnr(img.detach().cpu().numpy(), x_cvx.detach().cpu().numpy()))
        if iter==60:
            x_recon_gdc_60.append(x_cvx.detach().clone().cpu())


    data_loss = .5 * torch.norm(physics.A(x_cvx) - y) ** 2
    prior = lmbd * regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all.append(variational_loss.item())

    loss_gdc_all.append(loss_all)
    x_recon_gdc_300.append(x_cvx.detach().clone().cpu())
    psnr_gdc_all_gt.append(psnr_all_gt)
#%%
# GDD
lr_l2 = [50]
loss_gdd_all = []
x_recon_gdd_300 = []
x_recon_gdd_60 = []
psnr_gdd_all_gt = []
max_iter = 300

for lr in lr_l2:
    print(f'\nGDD: Initial lr={lr}')
    x_init = y.clamp(0,1).detach().clone()
    x_cvx = x_init.clone().detach().requires_grad_(True).to(device) 
    data_loss = .5 * torch.norm(physics.A(x_cvx) - y) ** 2
    prior = lmbd * regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all = []
    psnr_all = [compare_psnr(img.detach().cpu().numpy(), x_cvx.detach().cpu().numpy())]

    for iter in range(max_iter):
        alpha = lr/(1+iter)
        data_loss = .5 * torch.norm(physics.A(x_cvx) - y) ** 2
        prior = lmbd * regularizer.g(x_cvx)
        variational_loss = data_loss + prior
        grad = data_fidelity.grad(x_cvx, y, physics).detach() + lmbd * regularizer.grad(x_cvx).detach()
        x_cvx = x_cvx - alpha * grad
        recon_log = 'iter: {:d}/{:d}, var_loss: {:.6f}, data: {:.6f}, reg: {:.6f}'\
        .format(iter+1, max_iter, variational_loss.item(), data_loss.item(), prior.item())
        print(recon_log, end='\r')

        loss_all.append(variational_loss.item())
        psnr_all.append(compare_psnr(img.detach().cpu().numpy(), x_cvx.detach().cpu().numpy()))
        if iter==60:
            x_recon_gdd_60.append(x_cvx.detach().cpu())

    data_loss = .5 * torch.norm(physics.A(x_cvx) - y) ** 2
    prior = lmbd * regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all.append(variational_loss.item())

    x_recon_gdd_300.append(x_cvx.detach())
    loss_gdd_all.append(loss_all)
    psnr_gdd_all_gt.append(psnr_all)

# %%
# Visualize results
colors_l = [[0.37,0.77,0.79],[0.76,0.48,0.82],[0.93,0.69,0.13],[0.47,0.67,0.19],[0,0.45,0.74],[0.8500,0.33,0.1]]
xx = np.arange(len(loss_gdc_all[0]))
plt.semilogx(xx+1,loss_pd_all,label='Proposed')
plt.semilogx(xx+1,loss_gdc_all[0],label='SM-C: Step-size:{:.1f}'.format(lr_l[0]))
plt.semilogx(xx+1,loss_gdd_all[0],label='SM-D: Initial step-size:{:.1f}'.format(lr_l2[0]))
plt.semilogx(xx+1,stats['loss_all'],label='NMAPG')
plt.legend(loc='upper right',fontsize=14)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Objective value',fontsize=14)
plt.ylim([20,60])
plt.grid()
plt.show()

plt.semilogx(xx+1,psnr_pd_all_gt,label='Proposed')
plt.semilogx(xx+1,psnr_gdc_all_gt[0],label='SM-C: Step-size:{:.1f}'.format(lr_l[0]))
plt.semilogx(xx+1,psnr_gdd_all_gt[0],label='SM-D: Initial step-size:{:.1f}'.format(lr_l2[0]))
plt.semilogx(xx+1,stats['psnr_all'],label='NMAPG')
plt.legend(fontsize=14)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('PSNR',fontsize=14)
plt.ylim([11,32])
plt.grid()
plt.show()
#%%