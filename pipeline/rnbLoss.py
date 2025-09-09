from typing import Any, Dict, List, Optional, Tuple
from pipeline.attentionControl import AttentionStore
import torch
import torch.nn.functional as F
from pipeline.gaussion_smoothing import GaussianSmoothing
import torch.nn as nn
from config.boxLossConfig import boxConfig as config
from pipeline.LossUtil import LossUtil

smth_3 = GaussianSmoothing(sigma=3.0).cuda()


sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.bfloat16).cuda()

sobel_y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.bfloat16).cuda()

sobel_x = sobel_x.view(1, 1, 3, 3)
sobel_y = sobel_y.view(1, 1, 3, 3)

sobel_conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
sobel_conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)


sobel_conv_x.weight = nn.Parameter(sobel_x)
sobel_conv_y.weight = nn.Parameter(sobel_y)
def edge_loss_old(attn_map, mask, iou):
    
    loss_ = 0
    
    mask_clone = mask.clone()[1:-1, 1:-1]
    
    attn_map_clone = attn_map.unsqueeze(0).unsqueeze(0)
    attn_map_clone = attn_map_clone / attn_map_clone.max().detach()
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode='reflect')
    attn_map_clone = smth_3(attn_map_clone)

    sobel_output_x = sobel_conv_x(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_output_y = sobel_conv_y(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_sum = torch.sqrt(sobel_output_y ** 2  + sobel_output_x ** 2)
    sobel_sum = sobel_sum 
    
    loss_ += 1 - (sobel_sum * mask_clone).sum() / sobel_sum.sum() * (1 - iou)
    
    return loss_

def edge_loss(attn_map, mask, iou):
    loss_ = 0.0

    # 1. 处理掩码 (Mask)
    # 移除边缘，与后续 Sobel 输出对齐
    mask_clone = mask.clone()[1:-1, 1:-1]

    # 2. 处理注意力图 (Attention Map)
    # 增加批次和通道维度
    attn_map_clone = attn_map.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    
    # 归一化 - 关键：防止数值过大
    # 使用 detach() 避免梯度流回 max 操作
    max_val = attn_map_clone.max().detach()
    # 防止 max_val 为 0 或非常小导致除以 0
    if max_val < 1e-8:
        max_val = torch.tensor(1.0, device=max_val.device, dtype=max_val.dtype)
    attn_map_clone = attn_map_clone / max_val
    
    # 反射填充
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode='reflect')
    
    # 平滑 (假设 smth_3 是一个有效的平滑函数)
    attn_map_clone = smth_3(attn_map_clone) # 确保 smth_3 是数值稳定的

    # 3. 应用 Sobel 算子
    # 假设 sobel_conv_x 和 sobel_conv_y 是定义好的 nn.Conv2d 层
    sobel_output_x = sobel_conv_x(attn_map_clone).squeeze()[1:-1, 1:-1]  # [1, 1, H, W] -> [H-2, W-2]
    sobel_output_y = sobel_conv_y(attn_map_clone).squeeze()[1:-1, 1:-1]

    # 4. 计算边缘强度 - 核心修改点
    # 添加 eps 防止 sqrt(0) 和内部计算问题
    eps = 1e-8
    # 计算平方和，添加 eps
    grad_magnitude_sq = sobel_output_y ** 2 + sobel_output_x ** 2 + eps
    # 检查是否有 Inf 或 NaN (可选，用于调试)
    # if torch.isinf(grad_magnitude_sq).any() or torch.isnan(grad_magnitude_sq).any():
    #     print("Warning: Inf or NaN in grad_magnitude_sq")
    #     print("sobel_output_x range:", sobel_output_x.min().item(), sobel_output_x.max().item())
    #     print("sobel_output_y range:", sobel_output_y.min().item(), sobel_output_y.max().item())
    sobel_sum = torch.sqrt(grad_magnitude_sq) # sqrt(x + eps)

    # 5. 计算最终损失
    # 避免除以 0
    total_sobel = sobel_sum.sum()
    if total_sobel < 1e-8:
        total_sobel = torch.tensor(1.0, device=total_sobel.device, dtype=total_sobel.dtype)
    
    masked_sobel = (sobel_sum * mask_clone).sum()
    
    # 避免 (masked_sobel / total_sobel) > 1 导致 loss_ 为负 (虽然数学上可能，但通常不期望)
    # clamp 保证比值在 [0, 1] 范围内
    ratio = torch.clamp(masked_sobel / total_sobel, 0.0, 1.0)
    loss_ = 1.0 - ratio * (1.0 - iou)

    return loss_


def _compute_rnb_loss_whole(
    attention_store: AttentionStore,
    indices_to_alter:Dict[str, List[int]],
    gaussian_smoothing_kwargs:Dict[str, Any],
    shape: Tuple[int, int, int],
    bbox:List[List[int]],
    child_bbox:List[List[List[int]]],
    loss_util: LossUtil,
):
    # 解包参数
    smooth_attentions=gaussian_smoothing_kwargs.get("smooth_attentions", False)
    sigma=gaussian_smoothing_kwargs.get("sigma", 0.5)
    kernel_size=gaussian_smoothing_kwargs.get("kernel_size", 5)
    _, _, scale_factor = shape

    # 1.聚合所有attention maps
    """ Aggregates the attention across the different layers. """
    attention_texts = []
    attention_edges = []
    attention_maps = attention_store.get_store_attention()

    if "img-to-txt" in config.feature_map:
        for item in attention_maps[f"{'img-to-txt'}"]:
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[0]
            attention_texts.append(cross_maps)
            attention_edges.append(cross_maps)
    
    if "txt-to-img" in config.feature_map:
        for item in attention_maps[f"{'txt-to-img'}"]:
            item.transpose_(1, 2)
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[0]
            attention_texts.append(cross_maps)
            attention_edges.append(cross_maps)
    """ Aggregates the attention across the different layers. """
    attention_texts = torch.cat(attention_texts, dim=0)
    attention_texts = attention_texts.sum(0) / attention_texts.shape[0]

    attention_edges = torch.cat(attention_edges, dim=0)
    attention_edges = attention_edges.sum(0) / attention_edges.shape[0]
    attn_edge = torch.nn.functional.softmax(attention_edges[:, :, 1:] * 120, dim=-1) # 对 attn_edge 做 softmax（放大差异），用于后续边界感知

    # 2.解析position
    positions_list = [indices_to_alter[key] for key in indices_to_alter]

    obj_loss = 0
    loss = 0
    loss_list = []
    # 3.逐个box计算损失函数
    for cnt, indices in enumerate(positions_list):
        single_obj_loss = 0

        attn_list = []
        att_edge_list = []

        for i in indices:
            attn = attention_texts[:, :, i]
            attn_list.append(attn)

            edge = attn_edge[:, :, i]
            att_edge_list.append(edge)

        # attn_mean = torch.stack(attn_list).mean(dim=0)
        # attn_edge_map = torch.stack(att_edge_list).mean(dim=0)   # 在r&n中，此处是求和，而不是求平均

        attn_mean = torch.stack(attn_list).sum(dim=0)
        attn_edge_map = torch.stack(att_edge_list).sum(dim=0)
        # 3.1 基于极大极小值进行归一化
        attn_norm = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)

        # 3.2 计算box
        box = [max(round(b/scale_factor), 0) for b in bbox[cnt]]
        x1, y1, x2, y2 = box

        # 3.3 coordinates to masks
        obj_mask = torch.zeros_like(attn_norm)
        ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
        obj_mask[y1:y2, x1:x2] = ones_mask
        bg_mask = 1 - obj_mask

        # 3.4 动态阈值分割
        threshold = (attn_norm * obj_mask).sum() / obj_mask.sum() / 5 * 2 + \
            ((attn_norm * bg_mask).sum() / bg_mask.sum() / 5 * 3) if bg_mask.sum() != 0 else 0
        
        # print(attn_norm.sum())
        
        thres_image = attn_norm.gt(threshold) * 1.0 # thres_image：二值化后的前景 mask
        noise_image = F.sigmoid(20 * (attn_norm - threshold)) # noise_image：soft 版本，用于可导损失

        # print(thres_image.sum())

        # 3.5 提取预测的MBR（最小外接矩形）--> 提取的最小外接矩形存在为空的情况？也就是说可能存在没有值大于0.3的情况。
        # 这个0.3是怎么定的，此处需要更改？[此处需要DEBUG,待解决] -> 本质是梯度消失了？
        rows, cols = torch.where(thres_image > 0.3)
        x1, y1 = cols.min(), rows.min()
        x2, y2 = cols.max(), rows.max()

        mask_aug = torch.zeros_like(attn_norm)
        mask_aug[y1: y2, x1: x2] = 1    

        # 3.6 计算IOU
        mask_aug_in = mask_aug *  obj_mask  # 交
        iou = (mask_aug *  obj_mask).sum() / torch.max(mask_aug,  obj_mask).sum()

        if iou < 0.85:
            this_cls_diff_aug_1 = (mask_aug - attn_norm).detach() + attn_norm
            this_cls_diff_aug_in_1 = (mask_aug_in - attn_norm).detach() + attn_norm

            single_obj_loss += 1 - (1 - iou) * (obj_mask * this_cls_diff_aug_in_1).sum() * (1 / this_cls_diff_aug_1.sum().detach())
            single_obj_loss += 1 - (1 - iou) * (obj_mask * this_cls_diff_aug_in_1).sum().detach() * (1 / this_cls_diff_aug_1.sum())

            # if (attn_mean * obj_mask).max() < (attn_mean * bg_mask).max():
            #     obj_loss += edge_loss(attn_edge_map, obj_mask, iou) * 1 

            single_obj_loss += 1 - (1 - iou) * ((obj_mask * noise_image).sum() * (1 / noise_image.sum().detach())) * 0.5
            single_obj_loss += 1 - (1 - iou) * ((obj_mask * noise_image).sum().detach() * (1 / noise_image.sum())) * 0.5
    
        obj_loss += single_obj_loss
        loss_list.append(single_obj_loss)

    object_number = len(positions_list)
    loss = obj_loss / object_number

    print(f"loss:{loss}  loss_list:{loss_list}")

    name_list = config.text_index.keys()
    loss_dict = {name: loss_list[i] for i, name in enumerate(name_list)}

    if loss_util.is_initialized():
        loss = loss_util.compute_weighted_loss(loss_dict)
    else:
        loss_util.record_initial_losses(loss_dict)
    
    print(f"after reweight loss:{loss}")

    return loss, loss_list

def _compute_rnb_loss_character(
    attention_store: AttentionStore,
    indices_to_alter:Dict[str, List[int]],
    gaussian_smoothing_kwargs:Dict[str, Any],
    shape: Tuple[int, int, int],
    bbox:List[List[int]],
    child_bbox:List[List[List[int]]],
    loss_util: LossUtil,
):
    # 解包参数
    smooth_attentions=gaussian_smoothing_kwargs.get("smooth_attentions", False)
    sigma=gaussian_smoothing_kwargs.get("sigma", 0.5)
    kernel_size=gaussian_smoothing_kwargs.get("kernel_size", 5)
    _, _, scale_factor = shape

    # 1.聚合所有attention maps
    """ Aggregates the attention across the different layers. """
    attention_texts = []
    attention_edges = []
    attention_maps = attention_store.get_store_attention()

    if "img-to-txt" in config.feature_map:
        for item in attention_maps[f"{'img-to-txt'}"]:
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[0]
            attention_texts.append(cross_maps)
            attention_edges.append(cross_maps)
    
    if "txt-to-img" in config.feature_map:
        for item in attention_maps[f"{'txt-to-img'}"]:
            item.transpose_(1, 2)
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[0]
            attention_texts.append(cross_maps)
            attention_edges.append(cross_maps)
    """ Aggregates the attention across the different layers. """
    attention_texts = torch.cat(attention_texts, dim=0)
    attention_texts = attention_texts.sum(0) / attention_texts.shape[0]

    attention_edges = torch.cat(attention_edges, dim=0)
    attention_edges = attention_edges.sum(0) / attention_edges.shape[0]
    attn_edge = torch.nn.functional.softmax(attention_edges[:, :, 1:] * 120, dim=-1) # 对 attn_edge 做 softmax（放大差异），用于后续边界感知

    # 2.解析position
    positions_list = [indices_to_alter[key] for key in indices_to_alter]

    obj_loss = 0
    loss = 0
    loss_list = []
    # 3.逐个box计算损失函数
    for cnt, indices in enumerate(positions_list):
        single_obj_loss = 0

        character_bbox = child_bbox[cnt]

        for i in indices:
            attn = attention_texts[:, :, i]

            edge = attn_edge[:, :, i]

            attn_mean = attn
            attn_edge_map = edge

            # 3.1 基于极大极小值进行归一化
            attn_norm = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)

            # 3.2 计算box
            box = [max(round(b/scale_factor), 0) for b in bbox[cnt]]
            x1, y1, x2, y2 = box

            # 3.3 coordinates to masks
            obj_mask = torch.zeros_like(attn_norm)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            # 3.4 动态阈值分割
            threshold = (attn_norm * obj_mask).sum() / obj_mask.sum() / 5 * 2 + \
                ((attn_norm * bg_mask).sum() / bg_mask.sum() / 5 * 3) if bg_mask.sum() != 0 else 0
            
            # print(attn_norm.sum())
            
            thres_image = attn_norm.gt(threshold) * 1.0 # thres_image：二值化后的前景 mask
            noise_image = F.sigmoid(20 * (attn_norm - threshold)) # noise_image：soft 版本，用于可导损失

            # print(thres_image.sum())

            # 3.5 提取预测的MBR（最小外接矩形）--> 提取的最小外接矩形存在为空的情况？也就是说可能存在没有值大于0.3的情况。
            # 这个0.3是怎么定的，此处需要更改？[此处需要DEBUG,待解决] -> 本质是梯度消失了？
            rows, cols = torch.where(thres_image > 0.3)
            x1, y1 = cols.min(), rows.min()
            x2, y2 = cols.max(), rows.max()

            mask_aug = torch.zeros_like(attn_norm)
            mask_aug[y1: y2, x1: x2] = 1    

            # 3.6 计算IOU
            mask_aug_in = mask_aug *  obj_mask  # 交
            iou = (mask_aug *  obj_mask).sum() / torch.max(mask_aug,  obj_mask).sum()

            if iou < 0.85:
                this_cls_diff_aug_1 = (mask_aug - attn_norm).detach() + attn_norm
                this_cls_diff_aug_in_1 = (mask_aug_in - attn_norm).detach() + attn_norm

                single_obj_loss += 1 - (1 - iou) * (obj_mask * this_cls_diff_aug_in_1).sum() * (1 / this_cls_diff_aug_1.sum().detach())
                single_obj_loss += 1 - (1 - iou) * (obj_mask * this_cls_diff_aug_in_1).sum().detach() * (1 / this_cls_diff_aug_1.sum())

                # if (attn_mean * obj_mask).max() < (attn_mean * bg_mask).max():
                #     obj_loss += edge_loss(attn_edge_map, obj_mask, iou) * 1 

                single_obj_loss += 1 - (1 - iou) * ((obj_mask * noise_image).sum() * (1 / noise_image.sum().detach())) * 0.5
                single_obj_loss += 1 - (1 - iou) * ((obj_mask * noise_image).sum().detach() * (1 / noise_image.sum())) * 0.5
        
            obj_loss += single_obj_loss
            loss_list.append(single_obj_loss)

    object_number = len(positions_list)
    loss = obj_loss / object_number

    print(f"loss:{loss}  loss_list:{loss_list}")

    name_list = config.text_index.keys()
    loss_dict = {name: loss_list[i] for i, name in enumerate(name_list)}

    if loss_util.is_initialized():
        loss = loss_util.compute_weighted_loss(loss_dict)
    else:
        loss_util.record_initial_losses(loss_dict)
    
    print(f"after reweight loss:{loss}")

    return loss, loss_list

def compute_rnb_loss(
    attention_store: AttentionStore,
    indices_to_alter:Dict[str, List[int]],
    gaussian_smoothing_kwargs:Dict[str, Any],
    shape: Tuple[int, int, int],
    bbox:List[List[int]],
    child_bbox:List[List[List[int]]],
    loss_util: LossUtil,
):
    if child_bbox is None:
        return _compute_rnb_loss_whole(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
            shape=shape,
            bbox=bbox,
            child_bbox=child_bbox,
            loss_util=loss_util
        )
    else:
        return _compute_rnb_loss_character(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
            shape=shape,
            bbox=bbox,
            child_bbox=child_bbox,
            loss_util=loss_util
        )

