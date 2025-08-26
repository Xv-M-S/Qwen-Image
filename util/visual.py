import torch
import matplotlib.pyplot as plt
from config.boxLossConfig import boxConfig
from sklearn.decomposition import PCA
import os
import numpy as np
from matplotlib import font_manager
from typing import Optional
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipelineOutput

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)

# 设置中文字体支持
font_path = os.path.join(current_dir_path,'simhei.ttf')  # 替换为你的字体文件路径
prop = font_manager.FontProperties(fname=font_path)


# 绘制所有prompt的attention map特征平均值激活图
def visualize_feature_activation(feature, index):
    # feature: [1, 6032, 116]
    feature = feature.squeeze(0)  # -> [6032, 116]

    # 取 L2 范数（或绝对值最大）作为每个位置的激活强度
    activation = feature.norm(dim=-1)  # [6032]

    # 假设前 6032 是图像 patch，reshape 成 2D
    H, W = int(boxConfig.H /16), int(boxConfig.W/16)  
    img_tokens = activation[:int(H*W)]  # 截取图像部分
    activation_map = img_tokens.reshape(H, W).cpu().float().numpy()

    # 可视化
    plt.figure(figsize=(8, 8))
    plt.imshow(activation_map, cmap='viridis', interpolation='bilinear')
    plt.colorbar()
    plt.title("Feature Activation Map (L2 Norm)")
    plt.axis('off')

    # 保存图片到指定路径
    boxConfig.now_step = 0
    output_path = os.path.join(current_dir_path,f"../runing_output_tempfile/feature_map/feature_activation_map_all_{index}_{boxConfig.now_step}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 高分辨率保存

    plt.close()

# 绘制区域prompt的attention map的特征通道激活图
def visualize_feature_channel(feature, index):
    feature = feature.squeeze(0) 
    # Reshape 到图像
    H, W = int(boxConfig.H /16), int(boxConfig.W/16)  

    text_list = boxConfig.text_index

    # === 收集所有 key 的平均图 ===
    all_maps = []
    keys = []


    for key in text_list:
        channel_idx = text_list[key]
        all_channel_map = []
        for idx in channel_idx:
            try:
                channel_map = feature[:, idx].reshape(H, W)
                all_channel_map.append(channel_map)
            except Exception as e:
                print(f"Error processing key {key}")
                print(f"feature shape: {feature.shape}")
                continue
    
        channel_stack = torch.stack(all_channel_map)
        average_map = channel_stack.mean(dim=0).cpu().float().numpy()

        all_maps.append(average_map)
        keys.append(key)
    

    # === 计算网格布局 ===
    n = len(all_maps)
    ncols = int(np.ceil(np.sqrt(n)))   # 列数
    nrows = int(np.ceil(n / ncols))    # 行数

    # === 创建拼接图 ===
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()  # 统一处理为一维

    for idx, (avg_map, key) in enumerate(zip(all_maps, keys)):
        ax = axes[idx]
        im = ax.imshow(avg_map, cmap='viridis', interpolation='bilinear')
        ax.set_title(f"{key}", fontsize=12, fontproperties=prop)
        ax.axis('off')

        # 添加 colorbar（可选，太密集可去掉）
        # plt.colorbar(im, ax=ax, shrink=0.6)

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(current_dir_path,f"../runing_output_tempfile/feature_map/feature_activation_map_text_{index}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # print(f"✅ Saved feature map grid to {output_path}")


# 可视化去噪过程中的中间步骤的图像
def visualize_latent_map(pipe, latents, height, width, index, output_type: Optional[str] = "pil"):
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type=output_type)

    save_image = QwenImagePipelineOutput(images=image).images[0]

    # print(save_image)

    save_path = os.path.join(current_dir_path,f"../runing_output_tempfile/feature_map/middle_image_visual_{index}.png")
    save_image.save(save_path)