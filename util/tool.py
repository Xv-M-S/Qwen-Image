import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from config.boxLossConfig import boxConfig

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
# 设置中文字体支持
font_path = os.path.join(current_dir_path,'simhei.ttf')  # 替换为你的字体文件路径
prop = font_manager.FontProperties(fname=font_path)

# 计算函数运行时间
def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        if boxConfig.print_cost_time:
            print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun


# 可视化指定的输入layout
def visualize_mask_pairs(mask_pairs, width, height, output_path="visual_layout.png"):
    # 创建一个白色背景的图像
    fig, ax = plt.subplots(1, figsize=(width/100, height/100), dpi=100)
    ax.imshow(np.ones((height, width, 3)), aspect='equal')

    # 设置坐标轴不可见但保留绘图区域大小匹配图像尺寸
    ax.axis('off')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # y轴反向，匹配图像坐标系（左上角为原点）

    # 颜色和样式
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # 图例元素
    legend_elements = []

    for idx, (key, item) in enumerate(mask_pairs.items()):
        desc = item["description"]
        x1, y1, x2, y2 = item["mask"]
        w = x2 - x1
        h = y2 - y1
        
        color = colors[int(idx) % len(colors)]
        
        # 绘制矩形框
        rect = Rectangle((x1, y1), w, h, linewidth=3, 
                         edgecolor=color, facecolor='none', linestyle='-')
        ax.add_patch(rect)
        
        # 简化标签（避免太长）
        short_label = f"{key}: {desc.strip()}"
        if len(short_label) > 30:
            short_label = short_label[:27] + "..."
        
        # 在框上方添加文字（避免遮挡）
        text_y = y1 - 10 if y1 > height * 0.1 else y1 + h + 20  # 智能选择上下位置
        ax.text(x1, text_y, short_label, color=color, fontsize=9, fontproperties=prop,
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.7, edgecolor=color))

        # 添加图例条目
        legend_elements.append(Patch(
            edgecolor=color,
            facecolor='none',
            linewidth=2,
            label=f"Region {key}: {desc[:20]}..."
        ))

    # 添加图例
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, prop=prop)

    # 保存结果图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# 在生成的图像上可视化layout，看生成的图像是否符合layout的布局要求
def draw_masks_on_image(image_path, regional_prompt_mask_pairs, output_path='output_image.jpg'):
    # 打开图像
    image = Image.open(image_path)
    width, height = image.size  # 获取图像的宽高
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, 20)  # 尝试加载字体，可能需要根据你的系统调整路径
    except IOError:
        font = None  # 如果找不到字体文件，则不使用字体

    for key, value in regional_prompt_mask_pairs.items():
        mask = value['mask']
        description = value['description']

        # 计算矩形框的位置
        x1, y1, x2, y2 = mask
        text_position = (x1, y1 - 20) if y1 >= 20 else (x1, y2 + 5)  # 文本位置调整以避免超出图片边界

        # 绘制矩形框
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        # 在矩形框附近添加描述文本
        draw.text(text_position, description, fill="blue", font=font)

    # 保存结果
    image.save(output_path)
    print(f"标注完成，已保存至: {output_path}")

if __name__ == "__main__":
    regional_prompt_mask_pairs = {
        "0": {
            "description": ''' a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" ''',
            "mask": [128, 128, 384, 640]
        },
        "1": {
            "description": ''' a plaque sign "通义千问" ''',
            "mask": [500, 48, 756, 112]
        },
        "2": {
            "description": ''' a poster is written "π≈3.1415926-53589793-23846264-33832795-02384197" ''',
            "mask": [500, 640, 756, 780]
        }
    }

    CANVAS_WIDTH = 1664
    CANVAS_HEIGHT = 928
   
    # 调用函数显示
    # visualize_mask_pairs(regional_prompt_mask_pairs, CANVAS_WIDTH, CANVAS_HEIGHT)

    image_path = "/home/sxm/flux-workspace/Qwen-Image/example.png"
    # 可视化在图片上
    draw_masks_on_image(image_path=image_path, regional_prompt_mask_pairs=regional_prompt_mask_pairs)
