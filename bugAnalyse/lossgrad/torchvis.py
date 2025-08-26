import torch
import torch.nn as nn
from torchviz import make_dot

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 初始化模型
model = SimpleModel()

# 创建输入数据
x = torch.randn(1, 2)  # 输入维度为2的张量

# 执行前向传播
output = model(x)

# 使用make_dot生成计算图
dot = make_dot(output, params=dict(model.named_parameters()))

# 保存计算图为图片文件
dot.render("simple_model", format="png", cleanup=True)

print("计算图已生成并保存为'simple_model.png'")