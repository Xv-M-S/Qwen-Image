import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveBoxWeighting(nn.Module):
    def __init__(self, num_boxes, alpha=0.9, beta=0.95, epsilon=1e-8, gamma=0.1):
        """
        改进版：显式处理负下降率
        
        Args:
            gamma: 对负下降率的容忍阈值，若 delta < -gamma 才触发高权重
        """
        super().__init__()
        self.num_boxes = num_boxes
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma  # 负下降率触发阈值
        
        self.register_buffer('ema_loss', torch.zeros(num_boxes))
        self.register_buffer('ema_delta', torch.zeros(num_boxes))
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        
    def forward(self, box_losses):
        if self.training:
            with torch.no_grad():
                if self.step == 0:
                    self.ema_loss.copy_(box_losses)
                    self.ema_delta.fill_(0.0)
                    return box_losses, 0 , 0
                else:
                    # 计算瞬时下降量
                    delta_t = self.ema_loss - box_losses  # 正：下降；负：上升

                    # 计算瞬时下降比例
                    # delta_t = (self.ema_loss - box_losses) / (self.ema_loss + 1e-8)  # 正：下降比例；负：上升比例
                    
                    # 更新EMA下降率
                    self.ema_delta.mul_(self.beta).add_(delta_t, alpha=1 - self.beta)
                    
                    # 更新EMA损失
                    self.ema_loss.mul_(self.alpha).add_(box_losses, alpha=1 - self.alpha)
                self.step += 1
        
        # --- 关键：处理负下降率 ---
        ema_delta = self.ema_delta
        
        # 方法1：硬截断（强烈推荐用于负下降）
        # 所有负的 ema_delta 都视为“需紧急优化”
        # 权重 = 1 / (min(delta, 0) 的绝对值越大，权重越高)
        # 我们使用：w = 1 / max(delta, -gamma)  → 当 delta < -gamma，w 至少为 1/gamma
        # 把 ema_delta 中所有小于 -self.gamma 的值“截断”到 -self.gamma，从而防止负向冲击过大。
        # 方案A：偏移法（推荐）
        # safe_delta = torch.where(
        #     ema_delta < 0,
        #     torch.zeros_like(ema_delta),  # 将负值视为“零下降”
        #     ema_delta
        # )
        # weights = 1.0 / (safe_delta + self.epsilon)
        # 结果：负下降 → delta=0 → 权重极大（1/ε）

        # 方案B：指数映射（更平滑）
        # weights = torch.exp(-ema_delta)  # delta越小（负），exp(-delta)越大
        # 例如：delta = -0.1 → w ≈ 1.105；delta = 0.1 → w ≈ 0.905

        # 方案C：分段线性
        weights = torch.where(
            ema_delta < 0,
            10.0,  # 固定高权重
            1.0 / (ema_delta + self.epsilon)
        )
        
        # 可选：归一化权重
        # weights = weights / weights.mean()
        
        weighted_loss = (weights * box_losses).sum()
        
        return weighted_loss, weights.detach().cpu().numpy(), self.ema_delta.detach().cpu().numpy()




class AvgLossUtil:
    def __init__(self, num_boxes):
        self.AdaptiveBoxWeighting = AdaptiveBoxWeighting(num_boxes)
        self.lambda2 = 0.1

    
    def entropy_regularization_loss(self, box_losses, lambda_reg=0.1, alpha=1.0):
        """
        基于熵的均匀性正则项 Loss
        
        Args:
            box_losses: Tensor of shape (n_boxes,), 每个Box的损失值
            lambda_reg: 正则化系数 λ , 控制均匀性的重要性。
            alpha: 温度系数，控制分布锐度 , 控制分布的平滑程度（类似softmax中的温度）。
        
        Returns:
            total_loss: 标量，总损失
        """
        # 1. 原始任务损失（例如L1/L2/IoU等）
        task_loss = box_losses.sum()
        
        # 2. 构造损失的概率分布 p_i ∝ exp(-alpha * L_i)
        # 注意：我们希望损失小的Box权重高，所以用 -box_losses
        logits = -alpha * box_losses
        p = F.softmax(logits, dim=0)  # 归一化为概率分布
        
        # 3. 计算信息熵 H(p) = -sum(p_i * log(p_i))
        """
            衡量一个概率分布的 “不确定性” 或 “混乱程度”；
            值越大 → 分布越 均匀（最不确定）；
            值越小 → 分布越 尖锐（几乎确定某个类别/样本）。
        """
        entropy = -(p * torch.log(p + 1e-8)).sum()  # 加小数避免log(0)
        
        # 4. 负熵正则项（我们要最小化 -H，即最大化H）
        entropy_bonus = -entropy  # 注意：这是加到总损失中的“惩罚项”
        
        # 5. 总损失
        total_loss = task_loss + lambda_reg * entropy_bonus
        
        return total_loss, task_loss.item(), entropy.item()

    def uniform_box_loss(self, box_losses):
        # 归一化各Box损失（避免大损失Box主导）
        # box_losses = torch.stack(box_losses)
        # 这行代码把 box_losses 变成 均值为 0、标准差为 1 的相对分数，方便后续统一、稳定地比较或加权。
        normalized_losses = (box_losses - box_losses.mean()) / (box_losses.std() + 1e-8)

        total_loss = normalized_losses.sum() 
        return total_loss, normalized_losses

    def compute_loss(self, box_losses):
        # _, normalized_losses =self.uniform_box_loss(box_losses)

        total_loss, _, _ = self.entropy_regularization_loss(box_losses)

        weighted_loss, weights, delta = self.AdaptiveBoxWeighting(box_losses)

        final_loss = total_loss + self.lambda2 * weighted_loss

        return final_loss
    


if __name__ == '__main__':
    box_losses = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # box_losses = [torch.tensor(1.2), torch.tensor(0.9), torch.tensor(1.5)]
    avg_loss_util = AvgLossUtil(num_boxes=10)
    final_loss = avg_loss_util.compute_loss(box_losses)

    print(final_loss)