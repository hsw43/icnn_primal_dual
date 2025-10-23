##############################################################
#########################S&P##################################
##############################################################
#%%
# Set up 
import torch
import torch.nn.functional as F
from models import simple_ICNNPrior
from utils import power,proj_Lrelu,soft_thresh
import cv2
import skimage
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load image
img_size = 256
img = cv2.imread('data/snp.png')
img = skimage.color.rgb2gray(img)
img = torch.from_numpy(img).view(1,1,img_size,img_size).float().to(device)

# S&P noisy image
noise_prob = 0.25
torch.manual_seed(0)
probs = torch.rand(img.shape)
y = img.detach().clone().to(device)
y[probs<noise_prob/2] = 0
y[probs>1-noise_prob/2] = 1

# Regularization parameter
reg_param = 0.02

# Load regularizer
regularizer = simple_ICNNPrior(1,32,5,img_size,False,device)
ckp = torch.load('weights/reg_icnn_sp.pt')
regularizer.load_state_dict(ckp)

# Primal-Dual operators
def M0(u_in):
    return [regularizer.icnn.W0(u_in[0]), u_in[1]]

def M1(u_in):
    return [regularizer.icnn.WA(u_in[1])]

def M0T(v_in):
    return [regularizer.icnn.W0T(v_in[0]), v_in[1]]

def M1T(v_in):
    return [torch.zeros(1, 1, img_size, img_size).to(device), regularizer.icnn.WAT(v_in[0])]

x = y.detach().clone().to(device)
z = F.leaky_relu(regularizer.icnn.wx(x),0.2).detach().to(device)
v1 = [torch.zeros_like(z).to(device),torch.zeros_like(z).to(device)]
v2 = [torch.zeros_like(regularizer.icnn.WA(z)).to(device)]
v = [v1, v2]
xold = x.detach().clone().to(device)
zold = z.detach().clone().to(device)
u = [x,z]


W0norm = power(regularizer.icnn.W0, regularizer.icnn.W0T,u[0])
WAnorm = power(regularizer.icnn.WA, regularizer.icnn.WAT,u[1])

def MT_x(v_in):
    return regularizer.icnn.W0T(v_in[0][0])

def MT_z(v_in):
    return v_in[0][1] + regularizer.icnn.WAT(v_in[1][0])

#%%
# PD Test
c1_l = [0.005,0.01,0.05,0.1,0.5,1,5]
c2_l = [0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005]

max_iter = 200
b0 = regularizer.icnn.wx.bias.data.view(1,regularizer.icnn.n_filters,1,1)
bf1 = regularizer.icnn.fc1.bias.data
a_weight = regularizer.icnn.fc2.weight.data
cmin = 0*a_weight
cmax = a_weight
loss_pd_all = []
x_recon_pd_200 = []
x_recon_pd_15 = []
psnr_pd_all_gt = []

for c1 in c1_l:
    for c2 in c2_l:
        with torch.no_grad():
            print(f'\nPrimal- Dual: c1={c1}, c2={c2}')
            sigma0 = c1 / W0norm ** 2
            sigma1 = c2 / WAnorm ** 2
            taux = 1 / (sigma0 * W0norm ** 2)
            tauz = 1 / (sigma0 + sigma1 * WAnorm ** 2)

            x = y.clone().to(device)
            z = F.leaky_relu(regularizer.icnn.wx(x), 0.2)
            v1 = [torch.zeros_like(z), torch.zeros_like(z)]
            v_init = regularizer.icnn.fc2.weight.data.clone()
            v_init[regularizer.icnn.WA(z) + bf1 <= 0] = 0
            v2 = [v_init]
            v = [v1, v2]
            xold = x.clone()
            zold = z.clone()
            ubar = [2 * x - xold, 2 * z - zold]

            reg = regularizer.g(x)
            data_fid = reg_param * torch.norm(x-y,1)
            loss = data_fid+reg
            loss_all = [loss.item()]
            psnr_all_gt = [compare_psnr(img.detach().cpu().numpy(),x.detach().cpu().numpy())]

            for iter in range(max_iter):

                v_in1 = [a + sigma0 * b for a, b in zip(v[0], M0(ubar))]
                projv1 = proj_Lrelu([v_in1[0] / sigma0 + b0, v_in1[1] / sigma0], 0.2)
                v[0][0] = v_in1[0] - sigma0 * (projv1[0] - b0)
                v[0][1] = v_in1[1] - sigma0 * projv1[1] 
                v[1][0] = torch.clamp((v[1][0] + sigma1 * M1(ubar)[0]) + sigma1 * bf1, cmin, cmax)
                x = soft_thresh(x - taux * MT_x(v), y, taux * reg_param)
                z = z - tauz * MT_z(v)
                ubar = [2 * x - xold, 2 * z - zold]
                xold = x
                zold = z

                reg = regularizer.g(x)
                data_fid = reg_param * torch.norm(x - y, 1)
                loss = data_fid+reg
                loss_all.append(loss.item())
                psnr_all_gt.append(compare_psnr(img.detach().cpu().numpy(),x.detach().cpu().numpy()))
                recon_log = 'iter: {:d}/{:d},var_loss: {:.10f},fid: {:.10f},reg: {:.10f}'\
                        .format(iter+1, max_iter, loss.item(), data_fid.item(),reg.item())
                print(recon_log, end='\r')
                if iter==15:
                    x_recon_pd_15.append(x)

            x_recon_pd_200.append(x)

            loss_pd_all.append(loss_all)
            psnr_pd_all_gt.append(psnr_all_gt)

#%%
# GDC
lr_l = [0.1,0.5,1,2]
max_iter = 200
loss_gdc_all = []
x_recon_gdc_200 = []
x_recon_gdc_15 = []
psnr_gdc_all_gt = []

for lr in lr_l:
    print(f'\nGDC: lr={lr}')
    x_init = y.clamp(0,1).detach().clone().to(device)
    x_cvx = x_init.clone().detach().requires_grad_(True).to(device)

    data_loss = reg_param * torch.norm(x_cvx - y, 1)
    prior = regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    
    loss_all = []
    psnr_all_gt = [compare_psnr(img.detach().cpu().numpy(),x_cvx.detach().cpu().numpy())]
    
    for iter in np.arange(max_iter):
        data_loss = reg_param * torch.norm(x_cvx - y, 1)
        prior = regularizer.g(x_cvx)
        variational_loss = data_loss + prior
        grad = torch.autograd.grad(variational_loss,x_cvx)[0]
        x_cvx = x_cvx - lr * grad
        
        recon_log = 'iter: {:d}/{:d},var_loss: {:.6f},data: {:.6f},reg: {:.6f}'\
        .format(iter+1,max_iter,variational_loss.item(),data_loss.item(),prior.item())
        print(recon_log,end='\r')
        loss_all.append(variational_loss.item())
        psnr_all_gt.append(compare_psnr(img.detach().cpu().numpy(),x_cvx.detach().cpu().numpy()))
        if iter==15:
            x_recon_gdc_15.append(x_cvx.detach().clone().cpu())

    data_loss = reg_param * torch.norm(x_cvx - y, 1)
    prior = regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all.append(variational_loss.item())

    loss_gdc_all.append(loss_all)
    x_recon_gdc_200.append(x_cvx.detach().clone().cpu())
    psnr_gdc_all_gt.append(psnr_all_gt)
#%%
# GDD
lr_l2 = [1,3,5,10]
max_iter = 200
loss_gdd_all = []
x_recon_gdd_200 = []
x_recon_gdd_15 = []
psnr_gdd_all_gt = []

for lr in lr_l2:
    print(f'\nGDD: Initial lr={lr}')
    x_init = y.clamp(0,1).detach().clone().to(device)
    x_cvx = x_init.clone().detach().requires_grad_(True).to(device) 

    data_loss = reg_param * torch.norm(x_cvx - y, 1)
    prior = regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    
    loss_all = []
    psnr_all_gt = [compare_psnr(img.detach().cpu().numpy(),x_cvx.detach().cpu().numpy())]
    
    for iter in range(max_iter):
        alpha = lr/(1+iter)
        data_loss = reg_param * torch.norm(x_cvx - y, 1)
        prior = regularizer.g(x_cvx)
        variational_loss = data_loss + prior
        grad = torch.autograd.grad(variational_loss,x_cvx)[0]
        x_cvx = x_cvx-alpha*grad
        
        recon_log = 'iter: {:d}/{:d}\t,var_loss: {:.6f},data: {:.6f},reg: {:.6f}'\
        .format(iter+1,max_iter,variational_loss.item(),data_loss.item(),prior.item())
        print(recon_log,end='\r')
        loss_all.append(variational_loss.item())
        psnr_all_gt.append(compare_psnr(img.detach().cpu().numpy(),x_cvx.detach().cpu().numpy()))
        if iter==15:
            x_recon_gdd_15.append(x_cvx.detach().clone().cpu())

    data_loss = reg_param * torch.norm(x_cvx - y, 1)
    prior = regularizer.g(x_cvx)
    variational_loss = data_loss + prior
    loss_all.append(variational_loss.item())

    loss_gdd_all.append(loss_all)
    x_recon_gdd_200.append(x_cvx.detach().clone().cpu())
    psnr_gdd_all_gt.append(psnr_all_gt)

#%%
# Visualize results
# Ablation plots (AVG)
import numpy as np
import matplotlib.pyplot as plt

colors_l = [[0.37,0.77,0.79],[0.76,0.48,0.82],[0.93,0.69,0.13],[0.47,0.67,0.19],[0,0.45,0.74],[0.8500,0.33,0.1]]

loss_avg = []
for i in range(len(loss_pd_all)):
    loss_avg.append(sum(loss_pd_all[i])/max_iter)
Z = torch.asarray(loss_avg).view(7,7).numpy()
fig, ax = plt.subplots()
im = ax.imshow(Z.transpose())
ax.set_xticks(np.arange(0,7,1))
ax.set_yticks(np.arange(0,7,1))
ax.set_xticklabels(c1_l)
ax.set_yticklabels(c2_l)
ax.set_xlabel('$c_1$',fontsize=14)
ax.set_ylabel('$c_2$',fontsize=14)

plt.colorbar(im,ax=ax)
plt.show()

#%%
# Ablation plot (Energy)
cc_l = []
for c1 in c1_l:
    for c2 in c2_l:
        cc_l.append([c1,c2])

plt.semilogx(loss_pd_all[2],marker='s',markersize=8,markevery=0.2,color=colors_l[2],label='$c_1$=5e-3, $c_2$=5e-5')
plt.semilogx(loss_pd_all[44],marker='p',markersize=10,markevery=0.2,color=colors_l[3],label='$c_1$=5e-0, $c_2$=5e-5')
plt.semilogx(loss_pd_all[21],marker='H',markersize=10,markevery=0.2,color=colors_l[4],label='$c_1$=1e-1, $c_2$=5e-6')
plt.semilogx(loss_pd_all[27],marker='D',markersize=8,markevery=0.2,color=colors_l[5],label='$c_1$=1e-1, $c_2$=5e-3')
plt.semilogx(loss_pd_all[23],marker='*',markersize=12,markevery=[2,5,9,36,121],color=colors_l[1],label='$c_1$=1e-1, $c_2$=5e-5')
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Objective value',fontsize=14)
plt.legend(loc='upper right',prop={'size': 14},fancybox=True,framealpha=0.9)
plt.grid()
plt.show()
#%%
# Comp plots (GDC)
xx = np.arange(201)
plt.semilogx(xx+1,loss_pd_all[23],color=colors_l[1],label='Proposed')
for j in range(len(loss_gdc_all)):
    plt.semilogx(xx+1,loss_gdc_all[j],label='Step-size:{:.1f}'.format(lr_l[j]))
#plt.semilogx(x+1,loss_gdc_all[-1],color='#8c564b',label='Step-size:{:.1f}'.format(lr_l[-1]))
plt.legend(fontsize=12)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Objective value',fontsize=14)
plt.grid()
plt.show()

# Comp plots (GDD)
plt.semilogx(xx+1,loss_pd_all[23],color=colors_l[1],label='Proposed')
for j in range(len(loss_gdd_all)):
    plt.semilogx(xx+1,loss_gdd_all[j],label='Initial step-size:{:.1f}'.format(lr_l2[j]))
plt.legend(loc='upper right',fontsize=14)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Objective value',fontsize=14)
plt.grid()
plt.show()
# %%
# PSNR plots (GDC)
plt.semilogx(xx+1,psnr_pd_all_gt[23],color=colors_l[1],label='Proposed')
for j in range(len(psnr_gdc_all_gt)):
    plt.semilogx(xx+1,psnr_gdc_all_gt[j],label='Step-size:{:.1f}'.format(lr_l[j]))
#plt.plot(psnr_gdc_all_gt[-1],color='#8c564b',label='Step-size:{:.1f}'.format(lr_l[-1]))
plt.legend(fontsize=12)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('PSNR',fontsize=14)
plt.grid()
plt.show()

# PSNR plots (GDD)
plt.semilogx(xx+1,psnr_pd_all_gt[23],color=colors_l[1],label='Proposed')
for j in range(len(psnr_gdd_all_gt)):
    plt.semilogx(xx+1,psnr_gdd_all_gt[j],label='Initial step-size:{:.1f}'.format(lr_l2[j]))
plt.legend(fontsize=12)
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('PSNR',fontsize=14)
plt.grid()
plt.show()
# %%
# Recon
plt.imshow(x_recon_pd_15[23][0][0].clamp(0,1).detach().cpu(),cmap='gray')
plt.axis('off')
plt.show()
print('PD15:',compare_psnr(img.detach().cpu().numpy(),x_recon_pd_15[23].detach().cpu().numpy()))

plt.imshow(x_recon_pd_200[23][0][0].clamp(0,1).detach().cpu(),cmap='gray')
plt.axis('off')
plt.show()
print('PD200:',compare_psnr(img.detach().cpu().numpy(),x_recon_pd_200[23].detach().cpu().numpy()))

plt.imshow(x_recon_gdc_15[1][0][0].clamp(0,1).detach().cpu(),cmap='gray')
plt.axis('off')
plt.show()
print('GDC15:',compare_psnr(img.detach().cpu().numpy(),x_recon_gdc_15[1].detach().cpu().numpy()))

plt.imshow(x_recon_gdc_200[1][0][0].clamp(0,1).detach().cpu(),cmap='gray')
plt.axis('off')
plt.show()
print('GDC200:',compare_psnr(img.detach().cpu().numpy(),x_recon_gdc_200[1].detach().cpu().numpy()))

plt.imshow(x_recon_gdd_15[3][0][0].clamp(0,1).detach().cpu(),cmap='gray')
plt.axis('off')
plt.show()
print('GDD15:',compare_psnr(img.detach().cpu().numpy(),x_recon_gdd_15[3].detach().cpu().numpy()))

plt.imshow(x_recon_gdd_200[3][0][0].clamp(0,1).detach().cpu(),cmap='gray')
plt.axis('off')
plt.show()
print('GDD200:',compare_psnr(img.detach().cpu().numpy(),x_recon_gdd_200[3].detach().cpu().numpy()))