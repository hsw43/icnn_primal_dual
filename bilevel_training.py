# Bilevel training of ICNN with primal dual solver
# 
# Implementation adapted from:
# [1] https://github.com/johertrich/LearnedRegularizers

import torch
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from solver import PDHG_ICNN
import copy
import time
from utils import power

def estimate_lip(regularizer, dataset, device):
    if dataset is None:
        lip_max = 1.0
    else:
        with torch.no_grad():
            lip_avg = torch.tensor(0.0, device=device)
            lip_max = torch.tensor(0.0, device=device)
            for x in dataset:
                x = x['clean'].to(device)
                gradients = torch.sqrt(torch.sum(regularizer.grad(x) ** 2))
                lip_avg += gradients
                lip_max = torch.max(lip_max, gradients)
            lip_avg = lip_avg / len(dataset)

    return lip_max

def bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=100,
    PDHG_max_iter=1000,
    PDHG_tol_train=1e-4,
    PDHG_tol_val=1e-4,
    lr=0.005,
    lr_decay=0.99,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    validation_epochs=1,
    dynamic_range_psnr=False,
    upper_loss=lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1),
):
    
    momentum_optim = (0.9, 0.999)
    optimizer = torch.optim.Adam(
        regularizer.parameters(), lr=lr, betas=momentum_optim
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    x_power = torch.randn(1, 1, 256, 256, device=device)
    z_power = torch.randn(1, 32, 256, 256, device=device)
    with torch.no_grad():
        lip = estimate_lip(regularizer, train_dataloader, device)

        c1, c2 = .01, .001
        x_power = torch.randn(1, 1, 256, 256, device=device)
        z_power = torch.randn(1, 32, 256, 256, device=device)
        W0norm = power(regularizer.icnn.W0, regularizer.icnn.W0T, x_power)
        WAnorm = power(regularizer.icnn.WA, regularizer.icnn.WAT, z_power)

        sigma0 = c1 / W0norm ** 2
        sigma1 = c2 / WAnorm ** 2
        taux = 1 / (sigma0 * W0norm ** 2)
        tauz = 1 / (sigma0 + sigma1 * WAnorm ** 2)
        
    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    total_time = 0
    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
 
    for epoch in range(epochs):
        # ---- Training ----
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0

        train_step = 0
        iter_epoch = 0
        res_epoch = 0
        start_time_ = time.time()
        if epoch == 0:
            for x_gt in train_dataloader:
                img = x_gt['clean'].to(device).to(torch.float32)
                y = physics(img)
                x_noisy = physics.A_adjoint(y)
                train_loss_epoch += upper_loss(img, y).mean().item()
                train_psnr_epoch += psnr(y, img).mean().item()
            train_loss_epoch /= len(train_dataloader)
            train_psnr_epoch /= len(train_dataloader)
            
            val_loss_epoch = 0
            val_psnr_epoch = 0
            for x_val in val_dataloader:
                img_val = x_val['clean'].to(device).to(torch.float32)
                y_val = physics(img_val)
                x_val_noisy = physics.A_adjoint(y_val)

                val_loss_epoch += upper_loss(img_val, y_val).mean().item()
                val_psnr_epoch += psnr(y_val, img_val).mean().item()
            mean_val_loss = val_loss_epoch / len(val_dataloader)
            mean_val_psnr = val_psnr_epoch / len(val_dataloader)
            
            train_loss_epoch = 0
            train_psnr_epoch = 0

        for x in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            train_step += 1
            x_img = x['clean'].to(device).to(torch.float32)
            y = physics(x_img)
            x_noisy = physics.A_adjoint(y)

            
            x_recon, steps, res = PDHG_ICNN(
                x_noisy,
                y,
                x_img,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                sigma0,
                sigma1,
                taux,
                tauz,
                max_iter=PDHG_max_iter,
                tol=PDHG_tol_train,
                device=device,
                verbose=verbose,
            )
            
            optimizer.zero_grad()
            loss_fn = lambda x_in: upper_loss(x_img, x_in).mean()
            train_loss_epoch += loss_fn(x_recon).item()
            train_psnr_epoch += psnr(x_recon, x_img).mean().item()

            x_recon = x_recon.detach()

            if (train_step % 5) == 1:
                with torch.no_grad():
                    lip = estimate_lip(regularizer, train_dataloader, device)
                    
            grad = data_fidelity.grad(
                x_recon, y, physics
            ) + lmbd * regularizer.grad(x_recon)
            x_recon = x_recon - 1 / lip * grad
            loss = upper_loss(x_recon, x_img).mean()
            loss.backward()

            grad_params = [
                param.grad
                for param in optimizer.param_groups[0]["params"]
                if param.grad is not None
            ]
            
                
            optimizer.step()
            
            with torch.no_grad():
                W0norm = power(regularizer.icnn.W0, regularizer.icnn.W0T, x_power)
                WAnorm = power(regularizer.icnn.WA, regularizer.icnn.WAT, z_power)

                sigma0 = c1 / W0norm ** 2
                sigma1 = c2 / WAnorm ** 2
                taux = 1 / (sigma0 * W0norm ** 2)
                tauz = 1 / (sigma0 + sigma1 * WAnorm ** 2)
        
        scheduler.step()
        mean_train_loss = train_loss_epoch / len(train_dataloader)
        mean_train_psnr = train_psnr_epoch / len(train_dataloader)
        time_epoch = time.time() - start_time_
        total_time += time_epoch
        print(f"PDHG stop at step {iter_epoch/len(train_dataloader)} with residual {res_epoch/len(train_dataloader):.2E}")
   
        print_str = f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}"
        print(print_str)
        
        # ---- Validation ----
        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_psnr_epoch = 0
                for x_val in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val"
                ):
                    img_val = x_val['clean'].to(device).to(torch.float32)
                    y_val = physics(img_val)
                    x_val_noisy = physics.A_dagger(y_val)

                    x_recon_val, _, _ = PDHG_ICNN(
                        x_val_noisy,
                        y_val,
                        img_val,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd,
                        sigma0,
                        sigma1,
                        taux,
                        tauz,
                        max_iter=PDHG_max_iter,
                        tol=PDHG_tol_val,
                        device=device,
                        verbose=verbose,
                    )

                    val_loss_epoch += upper_loss(img_val, x_recon_val).mean().item()
                    val_psnr_epoch += psnr(x_recon_val, img_val).mean().item()

                mean_val_loss = val_loss_epoch / len(val_dataloader)
                mean_val_psnr = val_psnr_epoch / len(val_dataloader)
                print_str = f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.2E}, PSNR: {mean_val_psnr:.2f}"
                print(print_str)

                
                # ---- Save best regularizer based on validation PSNR ----
                if mean_val_psnr > best_val_psnr:
                    best_val_psnr = mean_val_psnr
                    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
                
    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)

    return regularizer
