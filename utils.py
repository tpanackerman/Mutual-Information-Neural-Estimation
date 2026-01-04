import torch
import numpy as np

def mine_loss(model, x, y):
    """
    Tính hàm loss MINE (Donsker-Varadhan bound).
    Output: loss (để tối ưu), mi_estimation (giá trị MI ước lượng)
    """
    # 1. Joint distribution (x, y đúng cặp)
    joint_t = model(x, y)
    
    # 2. Marginal distribution (x, y ghép ngẫu nhiên)
    y_shuffle = y[torch.randperm(y.shape[0])]
    marginal_t = model(x, y_shuffle)
    
    # 3. Tính Loss = -(Mean(T_joint) - Log(Mean(exp(T_marginal))))
    mi_est = torch.mean(joint_t) - torch.log(torch.mean(torch.exp(marginal_t)))
    loss = -mi_est
    
    return loss, mi_est

def true_mi_gaussian(rho, dim=1):
    """Tính MI lý thuyết cho trường hợp Gaussian"""
    return -0.5 * dim * np.log(1 - rho**2)