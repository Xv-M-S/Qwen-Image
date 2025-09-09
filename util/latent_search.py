#!/usr/bin/env python3
"""
给定矩阵 G，快速找到一张 randn 矩阵 R，使得 R 与 G 的经验分布最接近。
"""

import torch
import sys
from torch.distributions import Normal

def histogram_matching(z_opt, device, dtype):
    # 废弃：存在正负无穷值，且是经验逼近，存在失真
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

def quantile_transform(x, target_dist=Normal(0, 1)):
    """
    x: [B,C,H,W] 任意值
    return: 同 shape，经验分布近似 target_dist，且支持梯度
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, -1)                       # [B, N]
    N = x_flat.shape[-1]

    # 1. 经验秩 → 均匀 [0,1]
    #    torch.argsort 两次即可得到秩，完全可导
    rank = torch.argsort(torch.argsort(x_flat, dim=-1), dim=-1).float()
    uniform = (rank + 0.5) / N                   # 避免 0/1

    # 2. 均匀 → 目标分布分位
    #    Normal.icdf 支持广播，返回同 shape
    z_mapped = target_dist.icdf(uniform)

    # 3. reshape 回原空间
    return z_mapped.view(B, C, H, W)


def random_search(z_opt, target_dist=Normal(0, 1), max_iter=100000):
    # 废弃：随机搜索需要耗费太多时间
    """
    z_opt: [B,C,H,W] 随机值
    target_dist: 目标分布
    max_iter: 最大迭代次数
    return: 同 shape，经验分布近似 target_dist，且支持梯度
    """
    target = None
    now_loss = sys.maxsize
    for _ in range(max_iter):
        z_rand = torch.randn_like(z_opt)
        loss = torch.nn.functional.mse_loss(z_opt, z_rand)
        if loss < now_loss:
            target = z_rand
            now_loss = loss
            print("Iter:", _, "Loss:", loss.item())


def process_matrix_with_noise(matrix):
    # 重新排序只保留了“值集合”，却丢掉了“随机性/独立性”；因此它不再是正态分布矩阵。
    device = matrix.device
    dtype = matrix.dtype

    # 1. 展平矩阵
    flattened_matrix = matrix.flatten()

    # 2. 排序矩阵
    sorted_indices = torch.argsort(flattened_matrix)
    sorted_matrix = flattened_matrix[sorted_indices]

    # 3. 生成随机噪声
    noise = torch.randn(matrix.shape, device=device, dtype=dtype)  # 生成与原始矩阵形状相同的随机噪声

    print(f"noise mean:{noise.mean()}  noise std:{noise.std()}")

    # 4. 展平和排序噪声
    flattened_noise = noise.flatten()
    sorted_noise_indices = torch.argsort(flattened_noise)
    sorted_noise = flattened_noise[sorted_noise_indices]

    # 5. 根据原始矩阵的索引重新分布噪声
    rearranged_noise = torch.empty_like(flattened_noise, device=device, dtype=dtype)
    rearranged_noise[sorted_indices] = sorted_noise

    ret_noise = rearranged_noise.view(matrix.shape)

    print(f"ret noise mean:{ret_noise.mean()}  ret noise std:{ret_noise.std()}")

    return ret_noise
    
    


if __name__ == '__main__':
    # z_opt = torch.randn(1, 3, 64, 64)
    # device = z_opt.device
    # dtype = z_opt.dtype
    # z_matched = histogram_matching(z_opt, device, dtype)
    # mse = torch.nn.functional.mse_loss(z_opt, z_matched)
    # print("MSE(z_opt, z_matched) =", mse.item())
    # print(z_matched.shape)

    # 验证当N非常大时，会出现正负无穷的情况
    # import torch, math
    # N = 1000000000
    # q = torch.linspace(0.5/N, 1 - 0.5/N, N)
    # inf_mask = torch.isinf(torch.distributions.Normal(0,1).icdf(q))
    # print(inf_mask.any())          # True
    # print(torch.where(inf_mask))   # 最前/最后若干索引

    # 使用
    # z = torch.randn(2, 3, 64, 64, requires_grad=True)
    # z_norm = quantile_transform(z)
    # loss = z_norm.sum()
    # loss.requires_grad_(True)
    # loss.backward()  


    # z_opt = torch.randn(2, 3, 64, 64)
    # random_search(z_opt)
    pass