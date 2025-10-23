#%%
import torch
from bilevel_training import bilevel_training
from models import simple_ICNNPrior
from torch.utils.data import DataLoader, Subset
from dataset import inpaint_dataset
from deepinv.physics import Inpainting, GaussianNoise
from deepinv.optim import L2

device = 0 if torch.cuda.is_available() else 'cpu'

# Problem Setup
task = 'inpaint'
img_size = 256
noise_level = 0.03
mask = torch.load('data/mask.npy').to(device)
physics = Inpainting(mask.shape[1:],mask,device=device,noise_model=GaussianNoise(sigma=noise_level))
data_fidelity = L2(sigma=1.0)
lmbd = 0.1

# ICNN Prior Setup
smoothed = True #True for smoothed ICNN, False for non-smoothed ICNN
smoothing = 0.01 if smoothed else 0.0

regularizer = simple_ICNNPrior(1, 32, 5, img_size, smoothed, device)

# Dataset
train_dataset = inpaint_dataset(train=True)
val_ratio = 0.1
val_len = int(len(train_dataset) * val_ratio)
train_len = 160
train_set = Subset(train_dataset, range(train_len))
val_set = Subset(train_dataset, range(960, len(train_dataset)))
train_dataloader = DataLoader(train_set,batch_size=4,shuffle=True,drop_last=True)
val_dataloader = DataLoader(val_set,batch_size=1,shuffle=False,drop_last=True)
#%%
training_mode = 'BL-PD'
regularizer = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=30,
    PDHG_max_iter=500,
    PDHG_tol_train=1e-1,
    PDHG_tol_val=1e-1,
    lr=1e-3,
    lr_decay=0.99,
    validation_epochs=5,
    device=device,
    verbose=False   
)

model_path = 'reg_smoothed_icnn_ip_'+training_mode+'.pt' if smoothed else 'reg_icnn_ip_'+training_mode+'.pt'
torch.save(regularizer.state_dict(), model_path)
