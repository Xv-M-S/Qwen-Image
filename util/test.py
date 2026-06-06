import numpy as np
import torch
 
TEST = False

def process_matrix_with_noise(matrix):
    # 1. 展平矩阵
    flattened_matrix = matrix.flatten()

    # 2. 排序矩阵
    sorted_indices = torch.argsort(flattened_matrix)
    sorted_matrix = flattened_matrix[sorted_indices]

    # 3. 生成随机噪声
    noise = torch.rand(matrix.shape)  # 生成与原始矩阵形状相同的随机噪声
    if TEST:
        print("生成的随机噪声：")
        print(noise)

    # 4. 展平和排序噪声
    flattened_noise = noise.flatten()
    sorted_noise_indices = torch.argsort(flattened_noise)
    sorted_noise = flattened_noise[sorted_noise_indices]

    if TEST:
        print("排序的随机噪声：")
        print(sorted_noise)

    # 5. 根据原始矩阵的索引重新分布噪声
    rearranged_noise = torch.empty_like(flattened_noise)
    rearranged_noise[sorted_indices] = sorted_noise

    return sorted_matrix, rearranged_noise.view(matrix.shape)

if __name__ == "__main__":
    # 示例使用
    matrix = torch.tensor([[3, 1, 2],
                        [6, 5, 4]], dtype=torch.float32)

    sorted_matrix, rearranged_noise = process_matrix_with_noise(matrix)

    # 输出结果
    print("原始矩阵：")
    print(matrix)
    print("展平并排序后的矩阵：")
    print(sorted_matrix)
    print("重新分布的噪声：")
    print(rearranged_noise)

    """
    原始矩阵：
    [[3 1 2]
    [6 5 4]]
    展平并排序后的矩阵：
    [1 2 3 4 5 6]
    生成的随机噪声：
    [[0.25565946 0.86183132 0.80725691],
    [0.44934135 0.25036587 0.15312196]]
    展平并排序后的噪声：
    [0.15312196 0.25036587 0.25565946], 0.44934135 0.80725691 0.86183132]
    重新分布的噪声：
    [[0.25565946 0.15312196 0.25036587],
    [0.86183132 0.80725691 0.44934135]]


    """