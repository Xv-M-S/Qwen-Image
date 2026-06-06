from PIL import Image, ImageDraw, ImageFont

def find_optimal_font_size(draw, text, box_width, box_height, font_path):
    """使用二分查找找到适合指定区域的最大字体大小。"""
    low, high = 1, 500
    best_size = low

    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except OSError:
            # print(f"Font size {mid} failed to load.")
            high = mid - 1
            continue

        # --- 精确模拟换行 ---
        lines = []
        current_line = ""
        for char in text:
            test_line = current_line + char
            try:
                # Use textbbox for width calculation
                line_bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
            except Exception as e:
                # print(f"textbbox failed for line '{test_line}': {e}")
                line_width = len(test_line) * mid # Fallback

            if line_width <= box_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    # If even one char is too wide, put it on its own line
                    lines.append(char)
                    current_line = ""

        if current_line:
            lines.append(current_line)

        # --- Calculate total height based on bboxes ---
        if not lines:
            total_height = 0
        else:
            total_height = 0
            prev_line_bottom = 0
            for i, line in enumerate(lines):
                try:
                    line_bbox = draw.textbbox((0, 0), line, font=font)
                    line_top = line_bbox[1]
                    line_bottom = line_bbox[3]
                    line_height = line_bottom - line_top
                    
                    if i == 0:
                        # For the first line, its top might not be at y=0.
                        # We need to account for the space above the first line's top.
                        total_height += abs(line_top) # This is the ascent part for the first line
                    else:
                        # Add the gap between previous line's bottom and current line's top
                        leading = abs(line_top) # Space before current line's content
                        total_height += leading
                        
                    total_height += line_height
                    
                    if i < len(lines) - 1: # Add leading after this line, except the last
                        # A simple fixed leading, or could be proportional
                        total_height += line_height * 0.2 # Add 20% of line height as spacing
                        
                except Exception as e:
                    # print(f"textbbox failed for line '{line}': {e}")
                    # Fallback height calculation
                    total_height += mid 
                    if i < len(lines) - 1:
                         total_height += mid * 0.2 # Add fallback leading

        # --- Check if it fits ---
        # print(f"Font size {mid}: Calculated height {total_height}, Box height {box_height}")
        if total_height <= box_height:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1
            
    # print(f"Optimal font size found: {best_size}")
    return best_size

def render_text_in_box_with_visualization(image_width, image_height, text, box, font_path):
    """
    在指定的 box 内自适应字体大小并渲染多行文本，文本块居中对齐。
    返回每个字符的边界框列表和渲染后的图像。
    """
    if not text.strip():
        # Handle empty or whitespace-only text
        img = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline="blue", width=2)
        return [], img

    # 1. 创建临时图像和绘图对象用于字体大小计算
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    box_height = y_max - y_min

    # 2. 找到最优字体大小
    optimal_font_size = find_optimal_font_size(temp_draw, text, box_width, box_height, font_path)
    if optimal_font_size < 1:
        print("警告：无法找到合适的字体大小，使用最小字体 1。")
        optimal_font_size = 1

    # 3. 使用最优字体大小创建最终图像
    img = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, optimal_font_size)

    # 4. 再次进行换行（使用最终字体大小）
    lines = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        try:
            line_bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
        except Exception:
             line_width = len(test_line) * optimal_font_size

        if line_width <= box_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = char
            else:
                lines.append(char)
                current_line = ""
    if current_line:
        lines.append(current_line)

    if not lines:
         draw.rectangle(box, outline="blue", width=2)
         return [], img

    # 5. Calculate layout information for all lines to get precise total dimensions
    laid_out_lines = []
    y_cursor = 0
    max_line_width = 0
    for i, line in enumerate(lines):
        try:
            line_bbox = draw.textbbox((0, 0), line, font=font)
        except Exception:
             line_bbox = (0, 0, len(line) * optimal_font_size * 0.6, optimal_font_size)

        line_width = line_bbox[2] - line_bbox[0]
        line_height = line_bbox[3] - line_bbox[1]
        max_line_width = max(max_line_width, line_width)
        
        # Store layout info
        laid_out_lines.append({
            'text': line,
            'bbox': line_bbox,
            'width': line_width,
            'height': line_height,
            'y_top_offset': y_cursor - line_bbox[1] # Offset to align line's top to y_cursor
        })
        
        y_cursor += line_height
        # Add fixed leading between lines
        if i < len(lines) - 1:
            leading = max(1, int(line_height * 0.2)) # Ensure at least 1px leading
            y_cursor += leading

    total_text_block_height = y_cursor

    # --- Critical Fix 1: Ensure text block fits within box_height ---
    if total_text_block_height > box_height:
        print(f"Warning: Text block height ({total_text_block_height}) exceeds box height ({box_height}) after layout. Truncation or further shrinking might be needed. Proceeding with best effort.")
        # We can try to adjust, but font size finding should have prevented this.
        # For now, we proceed but might overflow slightly. 
        # A more robust solution would re-run font size search with stricter limits or implement truncation.

    # 6. Calculate offsets to center the entire text block within the box
    offset_x = x_min + max(0, (box_width - max_line_width) // 2)
    # Ensure offset_x doesn't cause overflow
    offset_x = min(offset_x, x_max - max_line_width)
    
    offset_y = y_min + max(0, (box_height - total_text_block_height) // 2)
    # Ensure offset_y doesn't cause overflow
    offset_y = min(offset_y, y_max - total_text_block_height)
    
    # --- Critical Fix 2: Adjust offset_y if the block is taller than the box (fallback) ---
    if total_text_block_height > box_height:
         offset_y = y_min # Align to top if it overflows


    # 7. 绘制文本和字符框
    char_boxes = []

    for line_info in laid_out_lines:
        line_text = line_info['text']
        line_bbox = line_info['bbox'] # bbox relative to (0,0)
        y_top_offset = line_info['y_top_offset'] # offset to align this line's top

        # Calculate the absolute top-left corner for drawing this line's text
        # line_bbox[0] accounts for potential negative left-side bearing
        line_draw_x = offset_x - line_bbox[0] 
        # y_top_offset places the line's top correctly, offset_y is the base offset for the whole block
        line_draw_y = offset_y + y_top_offset 

        # Draw each character and its bounding box
        char_cursor_x_relative = 0 # Cursor position relative to the start of the line
        for char in line_text:
            try:
                char_bbox_raw = draw.textbbox((0, 0), char, font=font)
            except Exception:
                 char_bbox_raw = (0, 0, optimal_font_size * 0.6, optimal_font_size)

            # Absolute position to draw the character
            char_draw_x = line_draw_x + char_cursor_x_relative
            char_draw_y = line_draw_y # line_draw_y already positions the line correctly

            # Absolute bounding box for the character on the canvas
            char_bbox_abs = (
                char_draw_x + char_bbox_raw[0],
                char_draw_y + char_bbox_raw[1],
                char_draw_x + char_bbox_raw[2],
                char_draw_y + char_bbox_raw[3]
            )

            char_boxes.append(char_bbox_abs)
            # Draw character bounding box
            draw.rectangle(char_bbox_abs, outline="red", width=1)
            # Draw character
            draw.text((char_draw_x, char_draw_y), char, font=font, fill="black")

            # Move cursor for next character
            char_width = char_bbox_raw[2] - char_bbox_raw[0]
            char_cursor_x_relative += char_width

    # 可选：绘制文本框边界
    draw.rectangle(box, outline="blue", width=2)

    return char_boxes, img




# --- 示例用法 ---
# 请确保将 'path/to/your/font.ttf' 替换为您系统上的有效字体文件路径
# 例如，在 Windows 上可能是 'C:/Windows/Fonts/simsun.ttc'
# 在 Linux 上可能是 '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
# 在 macOS 上可能是 '/System/Library/Fonts/PingFang.ttc'
if __name__ == "__main__":
    char_boxes, img = render_text_in_box_with_visualization(
        image_width=800,
        image_height=600,
        text="这是一段比较长的测试文本，用来验证自适应字体大小和多行换行功能是否正常工作。Happy Coding!",
        box=(50, 50, 750, 550), # (x_min, y_min, x_max, y_max)
        font_path="/home/sxm/flux-workspace/Qwen-Image/util/simhei.ttf" # <--- 请替换为实际字体路径
    )
    print("Character Boxes:", char_boxes)
    img.save("rendered_text.png")



