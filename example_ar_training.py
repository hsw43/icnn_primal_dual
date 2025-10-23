#%%
"""
Adversarial Training for ICNN

Subhadip Mukherjee and Sören Dittmer and Zakhar Shumaylov and Sebastian Lunz and Ozan Öktem and Carola-Bibiane Schönlieb
Learned convex regularizers for inverse problems

[1] https://arxiv.org/abs/2008.02839

Implementation adapted from:
[2] https://github.com/Subhadip-1/data_driven_convex_regularization
"""


import torch
from models import simple_ICNNPrior
from torch.utils.data import DataLoader
from dataset import inpaint_dataset

device = 0 if torch.cuda.is_available() else 'cpu'

img_size = 256

# ICNN Prior Setup
smoothed = True #True for smoothed ICNN, False for non-smoothed ICNN
smoothing = 0.01 if smoothed else 0.0
regularizer = simple_ICNNPrior(1, 32, 5, img_size, smoothed, device)

data_train = inpaint_dataset(train=True)
dataloader_train = DataLoader(data_train,batch_size=8,shuffle=True)
########### set-up optimizers ###########
optimizer = torch.optim.Adam(regularizer.parameters(), lr=5*1e-4, betas=(0.5, 0.99))

lambda_gp = 5.0 
n_epochs = 20

# Gradient penalty
def compute_gradient_penalty(regularizer, images_gt, images):
    real_samples = images_gt
    fake_samples = images

    B = real_samples.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=real_samples.device)
    interpolates = images_gt + alpha * (images - images_gt)
    interpolates.requires_grad_(True)
    grad_norm = regularizer.grad(interpolates).flatten(1).norm(2, dim=1)
    grad_loss = torch.nn.functional.relu(grad_norm - 1).square().mean()
    return grad_loss

#Training
regularizer.train()
loss_all = []


for epoch in range(n_epochs):
    total_loss, total_gp_loss, total_diff = 0,0,0
    for idx, data in enumerate(dataloader_train, 0):
        clean = (data['clean']).clamp(0,1).to(device)
        noisy = (data['noisy']).clamp(0,1).to(device)       
        
        #### training loss ####
        diff_loss = regularizer.g(clean).mean()-regularizer.g(noisy).mean()

        gp_loss = compute_gradient_penalty(regularizer, clean.data, noisy.data)
        loss = diff_loss + lambda_gp * gp_loss
        
        ####### parameter update #######
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_gp_loss += gp_loss.item()
        total_diff += diff_loss.item()

        regularizer.icnn.zero_clip_weights()
        
    ####### compute avg. loss over minibatches #######

    loss_all.append(total_loss/(idx+1))
                                    
    ######### save and print log ###########
    train_log = "epoch: [{}/{}], avg_loss: {:.8f}, avg_gradient_penalty: {:.8f}, avg_diff: {:.8f}".\
    format(epoch+1, n_epochs, total_loss/(idx+1), total_gp_loss/(idx+1), total_diff/(idx+1))
    print(train_log)

    acr_output = regularizer.g(clean)
            
    train_log = 'response to clean: ACR: {:.6f}'.format(torch.mean(acr_output))
    print(train_log)
            
    #### response to noisy images/FBP ####
    acr_output = regularizer.g(noisy)
            
    train_log = 'response to noisy: ACR: {:.6f}'.format(torch.mean(acr_output))
    print(train_log)
    
############# save the models #################
model_path = 'reg_smoothed_icnn_ip_ar.pt' if smoothed else 'reg_icnn_ip_ar.pt'
torch.save(regularizer.state_dict(), model_path)
# %%
