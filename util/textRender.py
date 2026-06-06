from PIL import Image, ImageDraw, ImageFont
import re

def create_sk_pairs(boxes, texts):
    """
    将 boxes 和 texts 合并为 sk_pairs 格式
    - 英文：按单词切分
    - 中文：按字符切分
    - 每个 token 分配一个 box（按顺序）
    """
    sk_pairs = {}
    index = 0
    tokens = []

    # 正则：匹配英文单词 或 中文字符
    pattern = re.compile(r'\b[a-zA-Z]+\b|[\u4e00-\u9fff]')

    # 第一步：从所有文本中提取 token（英文单词 + 中文字符）
    for text in texts:
        matches = pattern.findall(text)
        tokens.extend(matches)  # 每个匹配项是一个 token

    # 第二步：为每个 token 分配一个 box
    for token in tokens:
        if index >= len(boxes):
            print(f"警告: box 数量不足，token '{token}' 被忽略")
            break

        x, y, w, h = boxes[index]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)  # 转为 [x1, y1, x2, y2]

        sk_pairs[str(index)] = {
            "description": f'''"''' + token + '''"''',  # 等价于 '''"Qwen"'''
            "mask": [x1, y1, x2, y2]
        }
        index += 1

    return sk_pairs
def render_text_in_box_with_visualization(image_width, image_height, text, box, font_path, font_size):
    img = Image.new('RGB', (image_width, image_height), color='white')
    d = ImageDraw.Draw(img)
    
    font = ImageFont.truetype(font_path, font_size)
    
    x_min, y_min, x_max, y_max = box
    # 使用textsize获取整个字符串的大小，作为参考点
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    current_x = x_min + (x_max - x_min - text_width) / 2  # 居中对齐
    current_y = y_min
    
    char_boxes = []
    for char in text:
        char_bbox = d.textbbox((current_x, current_y), char, font=font)
        char_boxes.append((char_bbox[0], char_bbox[1], char_bbox[2], char_bbox[3]))
        d.rectangle(char_bbox, outline="red", width=2)  # 绘制字符的bounding box
        d.text((current_x, current_y), char, fill="black", font=font)
        current_x += char_bbox[2] - char_bbox[0]  # 更新x坐标为当前字符宽度
        
    return char_boxes, img

def render_text_with_custom_boxes(image_width, image_height, text, box, font_path, font_size):
    img = Image.new('RGB', (image_width, image_height), color='white')
    d = ImageDraw.Draw(img)
    
    font = ImageFont.truetype(font_path, font_size)
    
    x_min, y_min, x_max, y_max = box
    
    # 分词逻辑：英文按空格分，中文每个字一个 box
    words = []
    current_word = ''
    for char in text:
        if char == ' ':
            if current_word:
                words.append(current_word)
                current_word = ''
        elif '\u4e00' <= char <= '\u9fff':  # 中文字符
            if current_word:
                words.append(current_word)
                current_word = ''
            words.append(char)
        else:
            current_word += char
    if current_word:
        words.append(current_word)

    # 绘制参数
    current_x = x_min
    current_y = y_min
    max_height = 0

    char_boxes = []
    for word in words:
        # 关键：使用 anchor="lt" 确保 (current_x, current_y) 是左上角
        bbox = d.textbbox((current_x, current_y), word, font=font, anchor="lt")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        char_boxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))

        # 换行判断
        if current_x + w > x_max and current_x != x_min:
            current_x = x_min
            current_y += max_height + 5
            max_height = 0
            # 重新计算换行后的 bbox
            bbox = d.textbbox((current_x, current_y), word, font=font, anchor="lt")

        # 绘制红色边框（使用 bbox 坐标）
        d.rectangle(bbox, outline="red", width=2)
        # 绘制文本（使用左上角坐标）
        d.text((current_x, current_y), word, fill="black", font=font, anchor="lt")

        if h > max_height:
            max_height = h

        current_x += w + 5  # 单词间距

    return img, char_boxes

# 示例用法
# 存在局限性：只能渲染横的文本，只能渲染一行
# 扩展功能：添加竖直渲染，以及自适应box渲染
if __name__ == "__main__":
    # test function render_text_in_box_with_visualization
    # width, height = 500, 200
    # text_to_render = "Hello"
    # bounding_box = (50, 50, 450, 150)
    # font_path = "/home/sxm/flux-workspace/Qwen-Image/util/simhei.ttf"  # 替换为你的字体文件路径
    # font_size = 40
    
    # boxes, rendered_image = render_text_in_box_with_visualization(width, height, text_to_render, bounding_box, font_path, font_size)
    # print("Character Boxes:", boxes)
    # rendered_image.save("rendered_text.png")

    # test function render_text_with_custom_boxes
    # width, height = 600, 300
    # text_to_render = "a chalkboard sign 真牛"
    # bounding_box = (50, 50, 550, 250)
    # font_path = "/home/sxm/flux-workspace/Qwen-Image/util/simhei.ttf"  # 替换为你的字体文件路径
    # font_size = 40
    
    # rendered_image, char_boxes = render_text_with_custom_boxes(width, height, text_to_render, bounding_box, font_path, font_size)
    # print("Character Boxes:", char_boxes)
    # rendered_image.save("rendered_custom_text.png")


    # test function create_sk_pairs
    # 示例 boxes: [x, y, w, h]
    boxes = [
        [128, 240, 256, 400],   # Qwen
        [500, 48, 340, 112],    # 通
        [500, 160, 340, 112],   # 义
        [756, 640, 24, 140],    # 牛
        [800, 240, 200, 40],    # Coffee
    ]

    # 示例文本（每个文本可能包含多个 token）
    # test function create_sk_pairs
    texts = [
        "Qwen",
        "通义",
        "牛",
        "Coffee"
    ]

    result = create_sk_pairs(boxes, texts)
    # result = create_sk_pairs(char_boxes, text_to_render)

    # 打印结果
    print("sk_pairs = {")
    for k, v in result.items():
        print(f'    "{k}": {{')
        print(f'        "description": \'\'\'{v["description"]}\'\'\',')
        print(f'        "mask": {v["mask"]}')
        print(f'    }},')
    print("}")