import torch
from typing import Dict, List, Optional

class LossUtil:
    def __init__(self, loss_names: List[str], total_weight: float = 1.0):
        """
        损失函数工具类，用于动态调整损失权重

        Args:
            loss_names: 损失项名称列表，例如 ['loss_l1', 'loss_perceptual', 'loss_gan']
            total_weight: 所有损失权重之和（默认为1.0）
        """
        self.loss_names = loss_names
        self.total_weight = total_weight

        # 存储每个损失的初始值（在第一次调用时记录）
        self.initial_losses: Dict[str, float] = {}
        # 标记是否已初始化初始值
        self._is_initialized = False

    def record_initial_losses(self, losses: Dict[str, torch.Tensor]):
        """
        记录每个损失的初始值（通常在训练开始时调用一次）

        Args:
            losses: 当前各损失值的字典，key为名称，value为标量Tensor
        """
        for name in self.loss_names:
            if name not in losses:
                raise ValueError(f"Loss '{name}' not found in provided losses.")
            # 保存为Python float
            self.initial_losses[name] = losses[name].detach().item()
        self._is_initialized = True
        print("Initial losses recorded:", self.initial_losses)

    def is_initialized(self) -> bool:
        """检查是否已记录初始损失"""
        return self._is_initialized

    def compute_adaptive_weights(self, current_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        根据当前损失与初始值的变化，计算自适应权重

        Args:
            current_losses: 当前各损失值的字典

        Returns:
            weights: 每个损失对应的归一化权重（和为 total_weight）
        """
        if not self._is_initialized:
            raise RuntimeError("Initial losses not recorded. Call 'record_initial_losses' first.")

        # 计算每个损失相对于初始值的变化量（绝对差值）
        deltas = {}
        for name in self.loss_names:
            if name not in current_losses:
                raise ValueError(f"Current loss '{name}' not found.")
            current_val = current_losses[name].detach().item()
            initial_val = self.initial_losses[name]
            # 使用绝对差值，避免方向影响
            delta = abs(current_val - initial_val)
            deltas[name] = delta

        # 计算权重：变化越小，权重越大
        # 使用 1 / (1 + delta) 防止除零
        raw_weights = {name: 1.0 / (1.0 + delta) for name, delta in deltas.items()}

        # 归一化：使权重和为 total_weight
        sum_raw = sum(raw_weights.values())
        if sum_raw == 0:
            # 防止全为零（理论上不会发生）
            weights = {name: self.total_weight / len(self.loss_names) for name in self.loss_names}
        else:
            weights = {name: (raw_weight / sum_raw) * self.total_weight
                       for name, raw_weight in raw_weights.items()}

        return weights

    def compute_weighted_loss(self, current_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算加权总损失

        Args:
            current_losses: 当前各损失值的字典

        Returns:
            weighted_loss: 加权后的总损失（标量Tensor）
        """
        weights = self.compute_adaptive_weights(current_losses)
        print("Weights:", weights)
        weighted_sum = torch.tensor(0.0, device=next(iter(current_losses.values())).device)
        for name, weight in weights.items():
            weighted_sum += weight * current_losses[name]
        return weighted_sum

    def reset(self):
        """重置状态，可用于新训练阶段"""
        self.initial_losses.clear()
        self._is_initialized = False


if __name__ == "__main__":
    # 定义损失名称
    loss_names = ['loss_l1', 'loss_perceptual', 'loss_gan']

    # 创建工具类实例
    loss_util = LossUtil(loss_names, total_weight=1.0)

    # 模拟训练过程
    losses_t0 = {
        'loss_l1': torch.tensor(0.8),
        'loss_perceptual': torch.tensor(0.5),
        'loss_gan': torch.tensor(0.3)
    }

    # 第一步：记录初始损失（通常在第0个step）
    loss_util.record_initial_losses(losses_t0)

    # 后续step的损失
    losses_t1 = {
        'loss_l1': torch.tensor(0.75),        # 变化小 → 高权重
        'loss_perceptual': torch.tensor(0.2), # 变化大 → 低权重
        'loss_gan': torch.tensor(0.29)        # 变化极小 → 最高权重
    }

    # 自动计算权重
    weights = loss_util.compute_adaptive_weights(losses_t1)
    print("Adaptive weights:", weights)
    # 示例输出: {'loss_l1': 0.32, 'loss_perceptual': 0.18, 'loss_gan': 0.50} (sum=1.0)

    # 计算加权损失
    total_loss = loss_util.compute_weighted_loss(losses_t1)
    print("Weighted total loss:", total_loss.item())