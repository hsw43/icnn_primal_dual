import torch
import torch.nn.functional as F
from torch.nn.functional import leaky_relu
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

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
        o2_case1 = alpha * o1_case1

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
        o2 = torch.where(cond1, o2_case1,
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


def PDHG_ICNN(
        x0,
        y,
        x_gt,
        physics,
        data_fidelity,
        regularizer,
        lamda,
        sigma0,
        sigma1,
        taux,
        tauz,
        max_iter,
        tol,
        device,
        verbose=False,
        progress=False,
):
    
    """
    Proximal-Dual Hybrid Gradient (PDHG) algorithm for solving the optimization problem:
    
    min_x 0.5 * ||Ax - y||_2^2 + lmbd * reg(x)
    
    Parameters:
    - x0: Initial guess for the solution.
    - y: Observed data.
    - x_gt: Ground truth data (for evaluation).
    - mask: Inpainting mask
    - physics: Physics operator (object with A and A_adjoint methods).
    - data_fidelity: Data fidelity term (function).
    - lamda: Regularization parameter.
    - regularizer: Regularization (function).
    - sigma, tau: Step-sizes for the PDHG algorithm.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - device: Device to run the computation on (e.g., 'cpu' or 'cuda').
    - verbose: If True, print progress information.
    
    Returns:
    - x: The computed solution.
    """
    
    x = x0.detach().clone()
    z = regularizer.icnn.act2(regularizer.icnn.wx(x)).detach()     
    b0 = regularizer.icnn.wx.bias.data.view(1,regularizer.icnn.n_filters,1,1)
    bf1 = regularizer.icnn.fc1.bias.data
    a_weight = lamda*regularizer.icnn.fc2.weight.data
    omega = regularizer.icnn.smoothing
    cmin = 0*a_weight
    cmax = a_weight
    mu = 0
    regmu = lamda*mu

    if progress:
        logs = {
            'loss': [],
            'data_fid': [],
            'reg': [],
            'psnr': [],
            #'iter': 0
        }

    idx = torch.arange(0,x.shape[0],device=device)
    res = (tol+1)*torch.ones(x.shape[0],device=device)
    mask = physics.mask.data

    
    v11, v12 = torch.zeros_like(z), torch.zeros_like(z)
    # Expand v2 to have a batch dimension (batch size = 8)
    v2 = lamda * regularizer.icnn.fc2.weight.data.clone()
    v2 = v2.expand(x.shape[0], -1)  # (batch, ...)
    v2[regularizer.icnn.WA(z)+bf1<=0] = 0
    xold, zold = x, z
    xbar, zbar = x, z

    if progress:
        reg = lamda*regularizer.g(x).detach().squeeze(1)
        data_fid = data_fidelity(x,y,physics).detach() #.5*torch.norm(physics.A(x)-y)**2
        loss = data_fid+reg
        logs['loss'].append(loss.item())
        logs['data_fid'].append(data_fid.item())
        logs['reg'].append(reg.item())
        logs['psnr'].append(compare_psnr(x_gt.cpu().numpy(), x.cpu().numpy()))

    for iter in range(max_iter):
        with torch.no_grad():
            v_in1, v_in2 = v11[idx]+sigma0*regularizer.icnn.W0(xbar[idx]), v12[idx]+sigma0*zbar[idx]
            projv11, projv12= proj_epi(v_in1/sigma0+b0,v_in2/sigma0,regularizer)
            v11[idx], v12[idx] = v_in1-sigma0*(projv11-b0), v_in2-sigma0*projv12
            if regularizer.icnn.smoothed:
                v2[idx] = torch.clamp((a_weight*(v2[idx]+sigma1*(regularizer.icnn.WA(zbar[idx])+bf1)))/(sigma1*omega+a_weight),cmin,cmax) # huber
            else:
                v2[idx] = torch.clamp(v2[idx]+sigma1*regularizer.icnn.WA(zbar[idx])+sigma1*bf1,cmin,cmax)

            x[idx] = (x[idx]-taux*regularizer.icnn.W0T(v11[idx])+taux*physics.A_adjoint(y[idx]))/(1+taux*regmu+taux*mask)
            z[idx] = z[idx]-tauz*(v12[idx]+regularizer.icnn.WAT(v2[idx]))

            xbar[idx], zbar[idx] = 2*x[idx]-xold[idx], 2*z[idx]-zold[idx]
            xold[idx], zold[idx] = x[idx], z[idx]

        if iter > 0:
            grad = (data_fidelity.grad(x[idx], y[idx], physics) + lamda * regularizer.grad(x[idx])).detach()
            res[idx] = torch.norm(grad,p=2,dim=(1,2,3))

        condition = res >= tol
        idx = condition.nonzero().view(-1)

        if torch.max(res) < tol:
            if verbose:
                print('Convergence reached at iteration {:d}'.format(iter+1))
            break

            
        if progress:
            reg = lamda*(regularizer.g(x)).detach().squeeze(1)
            data_fid = data_fidelity(x,y,physics).detach()
            loss = data_fid+reg
            logs['loss'].append(loss.item())
            logs['data_fid'].append(data_fid.item())
            logs['reg'].append(reg.item())
            logs['psnr'].append(compare_psnr(x_gt.cpu().numpy(), x.cpu().numpy()))
            if verbose:
                recon_log = '[iter: {:d}/{:d},var_loss: {:.10f},fid: {:.10f},reg: {:.10f}]'\
                        .format(iter+1, max_iter, loss.item(), data_fid.item(),reg.item())
                print(recon_log)

    return (x, logs, iter+1, torch.max(res)) if progress else (x, iter+1, torch.max(res))

            