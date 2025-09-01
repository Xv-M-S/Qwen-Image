#!/usr/bin/env python3
"""
给定矩阵 G，快速找到一张 randn 矩阵 R，使得 R 与 G 的经验分布最接近。
"""

import torch
from torch.distributions import Normal

def histogram_matching(z_opt, device, dtype):
    # 展平
    z_flat = z_opt.flatten()
    N = z_flat.numel()
    
    # 对 z_opt 排序
    z_sorted, idx = torch.sort(z_flat)
    
    # 生成标准正态的分位数（理论值）
    q = torch.linspace(0.5/N, 1 - 0.5/N, N, device=device, dtype=dtype)
    normal_quantiles = Normal(0, 1).icdf(q)  # 标准正态的分位数
    # 根本原因：icdf(q) 在 q 接近 0 或 1 时会返回 ±∞（标准正态的理论分位在 q=0 时为 -∞，在 q=1 时为 +∞）。
    confidence = 0.99999        # 99.99 %
    c = Normal(0,1).icdf(torch.tensor([(1-confidence)/2])).abs().item()
    normal_quantiles = torch.clamp(normal_quantiles, -c, c)

    
    # 构建映射：z_sorted → normal_quantiles
    z_matched = torch.zeros_like(z_flat, device=device, dtype=dtype)
    z_matched[idx] = normal_quantiles  # 逆排序
    
    return z_matched.reshape(z_opt.shape)


if __name__ == '__main__':
    z_opt = torch.randn(1, 3, 64, 64)
    device = z_opt.device
    dtype = z_opt.dtype
    z_matched = histogram_matching(z_opt, device, dtype)
    mse = torch.nn.functional.mse_loss(z_opt, z_matched)
    print("MSE(z_opt, z_matched) =", mse.item())
    print(z_matched.shape)

    # 验证当N非常大时，会出现正负无穷的情况
    # import torch, math
    # N = 1000000000
    # q = torch.linspace(0.5/N, 1 - 0.5/N, N)
    # inf_mask = torch.isinf(torch.distributions.Normal(0,1).icdf(q))
    # print(inf_mask.any())          # True
    # print(torch.where(inf_mask))   # 最前/最后若干索引
