from PIL import Image
import os

# 图片路径列表（按顺序）
image_paths = [
    '/home/sxm/flux-workspace/Qwen-Image/inject10_baseRation0-1.png',
    '/home/sxm/flux-workspace/Qwen-Image/inject20_baseRation0-1.png',
    '/home/sxm/flux-workspace/Qwen-Image/inject30_baseRation0-1.png',
    '/home/sxm/flux-workspace/Qwen-Image/inject40_baseRation0-1.png',
    '/home/sxm/flux-workspace/Qwen-Image/inject50_baseRation0-1.png'
]

# 读取所有图片
images = []
for path in image_paths:
    if os.path.exists(path):
        img = Image.open(path)
        images.append(img)
    else:
        raise FileNotFoundError(f"图片未找到: {path}")

# 获取每张图片的尺寸
widths, heights = zip(*(img.size for img in images))

# 拼接后总高度 = 所有图片高度之和，宽度取最大值（或统一调整）
total_height = sum(heights)
max_width = max(widths)

# 创建一个空白图像，用于纵向拼接
concatenated_img = Image.new('RGB', (max_width, total_height))

# 纵向拼接：从上到下依次粘贴
y_offset = 0
for img in images:
    # 可选：将图片宽度调整为 max_width
    resized_img = img.resize((max_width, img.height), Image.Resampling.LANCZOS)
    concatenated_img.paste(resized_img, (0, y_offset))
    y_offset += resized_img.height

# 保存结果
concatenated_img.save('concatenated_vertical.jpg')
# concatenated_img.show()  # 可选：显示图片

print("纵向拼接完成，保存为 concatenated_vertical.jpg")