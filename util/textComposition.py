"""
给定一段文本，以及其中每个字符的box位置，我们需要将其中的属于同一个英文单词的box合并成一个大box！
输入示例：
"这是一段比较长的测试文本，用来验证自适应字体大小和多行换行功能是否正常工作。Happy Coding!"
[(53, 88, 130, 158), (130, 90, 207, 159), (207, 118, 284, 151), (284, 89, 361, 159), (361, 89, 438, 156), (438, 87, 515, 159), (515, 89, 592, 158), (592, 88, 669, 159), (669, 89, 746, 159), (53, 174, 130, 246), (130, 173, 207, 246), (207, 174, 284, 245), (284, 225, 361, 249), (361, 177, 438, 244), (438, 175, 515, 243), (515, 174, 592, 246), (592, 174, 669, 244), (669, 174, 746, 244), (53, 266, 130, 335), (130, 264, 207, 336), (207, 264, 284, 337), (284, 266, 361, 335), (361, 266, 438, 335), (438, 266, 515, 336), (515, 267, 592, 335), (592, 265, 669, 337), (669, 264, 746, 337), (53, 352, 130, 423), (130, 351, 207, 424), (207, 352, 284, 423), (284, 352, 361, 423), (361, 354, 438, 423), (438, 357, 515, 423), (515, 359, 592, 419), (592, 352, 669, 423), (669, 358, 746, 418), (53, 438, 130, 511), (130, 484, 207, 504), (207, 449, 246, 502), (246, 466, 285, 502), (285, 466, 324, 512), (324, 466, 363, 512), (363, 466, 402, 512), (402, 502, 441, 502), (441, 448, 480, 502), (480, 466, 519, 502), (519, 449, 558, 502), (558, 449, 597, 502), (597, 466, 636, 502), (636, 465, 675, 512), (675, 445, 714, 502)]
输出：
sk_pairs = {
    "0": {
        "description": '''"这"''',
        "mask": [53, 88, 130, 158]
    },
    ......
    "4": {
        "description": '''"Happy"''',
        "mask": [...]
    },
}
"""

import re
import json
from PIL import Image, ImageDraw, ImageFont
import random

def visualize_structured_boxes_with_text(image_width, image_height, structured_data, original_image_path=None, output_path="output_structured_boxes_with_text.png"):
    """
    在图像上可视化 structured_data 中的 mask 和 child_boxes，并添加文字描述。

    Args:
        image_width (int): 图像宽度。
        image_height (int): 图像高度。
        structured_data (dict): 包含 'description', 'mask', 'child_boxes' 的嵌套字典。
        original_image_path (str, optional): 原始图像的路径。如果提供，将在其上绘制。
                                             如果为 None，则创建一个白色背景的图像。
        output_path (str): 保存结果图像的路径。

    Returns:
        PIL.Image: 绘制了边界框和文字的图像对象。
    """
    # 决定背景图像
    if original_image_path:
        try:
            img = Image.open(original_image_path).convert('RGB')
            # 确保尺寸匹配或进行调整（这里假设尺寸是匹配的）
            if img.size != (image_width, image_height):
                print(f"警告：加载的图像尺寸 {img.size} 与提供的尺寸 ({image_width}, {image_height}) 不匹配，将调整大小。")
                img = img.resize((image_width, image_height))
        except FileNotFoundError:
            print(f"警告：未找到原始图像 {original_image_path}，将使用白色背景。")
            img = Image.new('RGB', (image_width, image_height), color='white')
    else:
        # 创建白色背景图像
        img = Image.new('RGB', (image_width, image_height), color='white')

    draw = ImageDraw.Draw(img)
    
    # 尝试加载一个默认字体，如果失败则使用默认字体（可能不支持中文）
    try:
        # 尝试使用系统上常见的字体，或者你可以指定一个字体文件的路径
        # font = ImageFont.truetype("arial.ttf", 15) # Windows
        # font = ImageFont.truetype("DejaVuSans.ttf", 15) # Linux
        font = ImageFont.truetype("PingFang.ttc", 15) # macOS / 或其他支持中文的字体
    except (OSError, IOError):
        print("警告：无法加载指定字体，将使用默认字体（可能不支持中文）。")
        # 使用默认字体，但指定大小
        font = ImageFont.load_default()
        # PIL 的默认字体不支持通过 size 参数改变大小，它是一个位图字体。
        # 如果需要更大或更清晰的默认文字，可能需要找到一个合适的 TTF 字体文件。

    # 定义颜色
    mask_color = "red"
    child_box_color = "blue"
    text_color = "black"
    # 为了让重叠的框更可见，可以使用半透明颜色，但这需要更复杂的处理（RGBA模式）
    # 这里我们使用不同的线宽
    mask_width = 3
    child_box_width = 1

    for key, item in structured_data.items():
        description = item.get('description', f'Item {key}')
        # 简化描述，只取引号内的内容或前几个词
        if description.startswith('"') and description.endswith('"'):
             simple_desc = description[1:-1] # 去掉首尾引号
        else:
             # 如果没有引号，取前N个字符作为标签
             simple_desc = description[:20] + "..." if len(description) > 20 else description

        mask = item.get('mask')
        child_boxes = item.get('child_boxes')

        # 绘制主 mask 框和文字
        if mask and len(mask) == 4:
            x1, y1, x2, y2 = mask
            draw.rectangle([x1, y1, x2, y2], outline=mask_color, width=mask_width)
            # 在左上角绘制文字
            text_position = (x1, y1 - 15) # 稍微向上偏移一点
            # 为了文字更清晰，可以添加背景或描边，这里简单绘制
            draw.text(text_position, simple_desc, fill=text_color, font=font)
        else:
             print(f"警告：项目 {key} 的 mask 无效或缺失，跳过。")

        # 绘制 child_boxes 和索引文字
        if child_boxes:
            for i, box in enumerate(child_boxes):
                if box and len(box) == 4:
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline=child_box_color, width=child_box_width)
                    # 在左上角绘制子框索引
                    text_position = (x1, y1)
                    draw.text(text_position, str(i), fill=text_color, font=font)
                else:
                    print(f"警告：项目 {key} 的 child_boxes[{i}] 无效，跳过。")
        else:
             print(f"信息：项目 {key} 没有 child_boxes。")


    # 保存图像
    try:
        img.save(output_path)
        print(f"图像已保存到 {output_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    return img


def visualize_boxes(image_width, image_height, sk_pairs, original_image_path=None, output_path="output_with_boxes.png"):
    """
    在图像上可视化 sk_pairs 中的边界框。

    Args:
        image_width (int): 图像宽度。
        image_height (int): 图像高度。
        sk_pairs (dict): 包含 'description' 和 'mask' 的字典。
        original_image_path (str, optional): 原始图像的路径。如果提供，将在其上绘制。
                                             如果为 None，则创建一个白色背景的图像。
        output_path (str): 保存结果图像的路径。

    Returns:
        PIL.Image: 绘制了边界框的图像对象。
    """
    # 决定背景图像
    if original_image_path:
        try:
            img = Image.open(original_image_path).convert('RGB')
            # 确保尺寸匹配或进行调整（这里假设尺寸是匹配的）
            if img.size != (image_width, image_height):
                print(f"警告：加载的图像尺寸 {img.size} 与提供的尺寸 ({image_width}, {image_height}) 不匹配，将调整大小。")
                img = img.resize((image_width, image_height))
        except FileNotFoundError:
            print(f"警告：未找到原始图像 {original_image_path}，将使用白色背景。")
            img = Image.new('RGB', (image_width, image_height), color='white')
    else:
        # 创建白色背景图像
        img = Image.new('RGB', (image_width, image_height), color='white')

    draw = ImageDraw.Draw(img)
    
    # （可选）加载字体用于绘制标签
    try:
        # 尝试加载一个系统字体，你可以替换为你系统上的字体路径
        # 例如: font = ImageFont.truetype("arial.ttf", 15)
        # 或者使用默认字体（可能较小）
        font = ImageFont.load_default() 
    except:
        font = ImageFont.load_default()
        print("无法加载指定字体，使用默认字体。")


    # 为不同的项目类型定义颜色
    single_char_color = "blue"
    word_color = "red"
    space_color = "green" # 可选：为空格使用不同颜色

    for key, item in sk_pairs.items():
        description = item.get('description', 'N/A')
        mask = item.get('mask')
        
        if not mask or len(mask) != 4:
            print(f"警告：项目 {key} 的 mask 无效或缺失，跳过。")
            continue

        x1, y1, x2, y2 = mask

        # 确定颜色
        # 移除 description 中的引号进行判断
        clean_desc = description.strip('"')
        if len(clean_desc) == 1 and clean_desc != ' ':
            color = single_char_color
            label = key # 可以显示索引
        elif clean_desc == ' ':
            color = space_color
            label = f"{key}:{description}" # 显示索引和空格描述
        else: # 单词
            color = word_color
            label = f"{key}:{description}" # 显示索引和单词描述

        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # （可选）绘制标签
        try:
            # 获取文本大小以进行定位（对于默认字体可能不精确）
            # left, top, right, bottom = draw.textbbox((0,0), label, font=font)
            # text_width = right - left
            # text_height = bottom - top
            # 简化处理，假设一个大概的大小
            text_width = len(label) * 6 # 粗略估计
            text_height = 12 # 粗略估计
            
            # 将标签放在框的左上角内侧
            label_x = x1 + 2
            label_y = y1 + 2
            
            # 确保标签在图像边界内
            label_x = max(0, min(label_x, image_width - text_width))
            label_y = max(0, min(label_y, image_height - text_height))
            
            # 绘制背景以提高标签可读性 (可选)
            # draw.rectangle([label_x, label_y, label_x + text_width, label_y + text_height], fill="white")
            
            draw.text((label_x, label_y), label, fill=color, font=font)
        except Exception as e:
            # 如果绘制标签出错（如字体问题），则跳过
            print(f"绘制标签 '{label}' 时出错: {e}")

    # 保存图像
    try:
        img.save(output_path)
        print(f"图像已保存到 {output_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    return img

def merge_english_word_boxes(text, char_boxes):
    """
    遍历文本和字符boxes，将连续的英文单词字符的boxes合并。

    Args:
        text (str): 输入的完整文本。
        char_boxes (list of tuples): 每个字符对应的边界框 (x1, y1, x2, y2)。

    Returns:
        dict: 包含合并后信息的字典，格式如示例所示。
    """
    if len(text) != len(char_boxes):
        raise ValueError("文本长度与边界框列表长度不匹配。")

    sk_pairs = {}
    item_index = 0
    i = 0
    n = len(text)

    current_word_chars = []
    current_word_boxes = []

    while i < n:
        char = text[i]
        box = char_boxes[i]

        # 检查当前字符是否为英文字母
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            # 是英文字符，加入当前单词
            current_word_chars.append(char)
            current_word_boxes.append(box)
        else:
            # 不是英文字符，处理当前可能存在的单词
            if current_word_chars:
                # 合并单词的boxes
                merged_box = merge_boxes(current_word_boxes)
                word_str = "".join(current_word_chars)
                
                sk_pairs[str(item_index)] = {
                    "description": f'"{word_str}"',
                    "mask": list(merged_box) # 转换为列表以便JSON序列化
                }
                item_index += 1
                
                # 重置单词缓存
                current_word_chars = []
                current_word_boxes = []
            
            # 处理当前非英文字符
            if char != ' ':
                sk_pairs[str(item_index)] = {
                    "description": f'"{char}"',
                    "mask": list(box) # 转换为列表
                }
            item_index += 1

        i += 1

    # 循环结束后，检查是否还有未处理的单词（例如文本以英文单词结尾）
    if current_word_chars:
        merged_box = merge_boxes(current_word_boxes)
        word_str = "".join(current_word_chars)
        sk_pairs[str(item_index)] = {
            "description": f'"{word_str}"',
            "mask": list(merged_box)
        }

    return sk_pairs

def merge_boxes(boxes):
    """
    将多个边界框合并成包围所有框的最小矩形框。

    Args:
        boxes (list of tuples): 边界框列表 [(x1, y1, x2, y2), ...]

    Returns:
        tuple: 合并后的边界框 (min_x1, min_y1, max_x2, max_y2)
    """
    if not boxes:
        return (0, 0, 0, 0) # 或者抛出异常
        
    # 初始化为第一个框的值
    min_x1 = boxes[0][0]
    min_y1 = boxes[0][1]
    max_x2 = boxes[0][2]
    max_y2 = boxes[0][3]

    # 遍历剩余的框，更新最值
    for box in boxes[1:]:
        x1, y1, x2, y2 = box
        if x1 < min_x1:
            min_x1 = x1
        if y1 < min_y1:
            min_y1 = y1
        if x2 > max_x2:
            max_x2 = x2
        if y2 > max_y2:
            max_y2 = y2
            
    return (min_x1, min_y1, max_x2, max_y2)


if __name__ == "__main__":
    # --- 示例输入 ---
    text_input = "这是一段比较长的测试文本，用来验证自适应字体大小和多行换行功能是否正常工作。Happy Coding!"
    char_boxes_input = [
        (53, 88, 130, 158), (130, 90, 207, 159), (207, 118, 284, 151), (284, 89, 361, 159),
        (361, 89, 438, 156), (438, 87, 515, 159), (515, 89, 592, 158), (592, 88, 669, 159),
        (669, 89, 746, 159), (53, 174, 130, 246), (130, 173, 207, 246), (207, 174, 284, 245),
        (284, 225, 361, 249), (361, 177, 438, 244), (438, 175, 515, 243), (515, 174, 592, 246),
        (592, 174, 669, 244), (669, 174, 746, 244), (53, 266, 130, 335), (130, 264, 207, 336),
        (207, 264, 284, 337), (284, 266, 361, 335), (361, 266, 438, 335), (438, 266, 515, 336),
        (515, 267, 592, 335), (592, 265, 669, 337), (669, 264, 746, 337), (53, 352, 130, 423),
        (130, 351, 207, 424), (207, 352, 284, 423), (284, 352, 361, 423), (361, 354, 438, 423),
        (438, 357, 515, 423), (515, 359, 592, 419), (592, 352, 669, 423), (669, 358, 746, 418),
        (53, 438, 130, 511), (130, 484, 207, 504), (207, 449, 246, 502), (246, 466, 285, 502),
        (285, 466, 324, 512), (324, 466, 363, 512), (363, 466, 402, 512), (402, 502, 441, 502),
        (441, 448, 480, 502), (480, 466, 519, 502), (519, 449, 558, 502), (558, 449, 597, 502),
        (597, 466, 636, 502), (636, 465, 675, 512), (675, 445, 714, 502)
    ]

    # --- 执行函数 ---
    result_sk_pairs = merge_english_word_boxes(text_input, char_boxes_input)

    # --- 打印结果 (格式化为JSON以便查看) ---
    # print(json.dumps(result_sk_pairs, indent=4, ensure_ascii=False))

    # --- 或者直接打印字典 ---
    print(result_sk_pairs)

    # 图像尺寸 (需要与生成 boxes 时的图像尺寸一致)
    img_width = 800
    img_height = 600

    # 如果你有原始图像，可以提供路径
    # original_img_path = "your_original_image.png"

    # 调用函数进行可视化
    # 如果没有原始图像，会创建一个白色背景的图像
    visualized_img = visualize_boxes(
        image_width=img_width,
        image_height=img_height,
        sk_pairs=result_sk_pairs,
        original_image_path=None, # 替换为你的原始图像路径，或保持 None
        output_path="visualized_boxes.png"
    )

    # 如果在支持显示的环境中（如Jupyter Notebook），可以直接显示
    # visualized_img.show()



