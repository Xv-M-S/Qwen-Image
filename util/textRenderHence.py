from PIL import Image, ImageDraw, ImageFont

def find_optimal_font_size(draw, text, box_width, box_height, font_path, direction='horizontal'):
    """使用二分查找找到适合指定区域的最大字体大小。"""
    low, high = 1, 500
    best_size = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except OSError:
            high = mid - 1
            continue

        if direction == 'horizontal':
            # --- 横向布局逻辑 ---
            lines = []
            current_line = ""
            for char in text:
                test_line = current_line + char
                try:
                    line_bbox = draw.textbbox((0, 0), test_line, font=font)
                    line_width = line_bbox[2] - line_bbox[0]
                except Exception: 
                    line_width = len(test_line) * mid

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
                total_height = 0
            else:
                total_height = 0
                for i, line in enumerate(lines):
                    try:
                        line_bbox = draw.textbbox((0, 0), line, font=font)
                        line_height = line_bbox[3] - line_bbox[1]
                    except Exception:
                        line_height = mid

                    if i == 0:
                        total_height += abs(line_bbox[1]) # ascent part for first line
                    else:
                        total_height += abs(line_bbox[1]) # leading before line

                    total_height += line_height

                    if i < len(lines) - 1:
                        total_height += line_height * 0.2 # Add leading

        elif direction == 'vertical':
            # --- 竖向布局逻辑 ---
            columns = []
            current_column = ""
            for char in text:
                test_column = current_column + char
                # For vertical, we check if the "column" (list of chars) fits in height
                temp_total_height = 0
                temp_line_heights = []
                for c in test_column:
                    try:
                        char_bbox = draw.textbbox((0, 0), c, font=font)
                        char_height = char_bbox[3] - char_bbox[1]
                        temp_line_heights.append(char_height)
                    except Exception:
                        char_height = mid
                        temp_line_heights.append(char_height)
                    temp_total_height += char_height + (char_height * 0.2 if len(columns) > 0 or len(current_column) > 0 else 0) # Simplified leading

                # Subtract the last added leading for check
                if temp_line_heights:
                    temp_total_height -= temp_line_heights[-1] * 0.2

                if temp_total_height <= box_height:
                    current_column = test_column
                else:
                    if current_column:
                        columns.append(current_column)
                        current_column = char
                    else:
                        columns.append(char)
                        current_column = ""
            if current_column:
                columns.append(current_column)

            # Calculate total width needed for columns
            if not columns:
                total_height = 0 # Here, total_height represents total_width for vertical
            else:
                max_column_height = 0
                total_width = 0
                column_widths = []
                for i, col in enumerate(columns):
                    col_width = 0
                    col_height = 0
                    for j, char in enumerate(col):
                        try:
                            char_bbox = draw.textbbox((0, 0), char, font=font)
                            char_width = char_bbox[2] - char_bbox[0]
                            char_height = char_bbox[3] - char_bbox[1]
                            col_width = max(col_width, char_width)
                            col_height += char_height + (char_height * 0.2 if j < len(col) - 1 else 0)
                        except Exception:
                             col_width = max(col_width, mid * 0.6)
                             char_height = mid
                             col_height += char_height + (char_height * 0.2 if j < len(col) - 1 else 0)
                    column_widths.append(col_width)
                    max_column_height = max(max_column_height, col_height)
                    total_width += col_width + (col_width * 0.1 if i < len(columns) - 1 else 0) # Add column spacing

                total_height = total_width # For vertical, we check width against box_width

        # Check if it fits
        fits = False
        if direction == 'horizontal':
            fits = (total_height <= box_height)
        elif direction == 'vertical':
             # For vertical, total_height represents the calculated width
             fits = (total_height <= box_width)

        if fits:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_size

def render_text_in_box_with_visualization(image_width, image_height, text, box, font_path, direction='horizontal'):
    """
    在指定的 box 内自适应字体大小并渲染多行文本，支持横向和竖向，文本块居中对齐。
    返回每个字符的边界框列表和渲染后的图像。
    """
    if not text.strip():
        img = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline="blue", width=2)
        return [], img

    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    box_height = y_max - y_min

    optimal_font_size = find_optimal_font_size(temp_draw, text, box_width, box_height, font_path, direction)
    if optimal_font_size < 1:
        print("警告：无法找到合适的字体大小，使用最小字体 1。")
        optimal_font_size = 1

    img = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, optimal_font_size)

    char_boxes = []

    if direction == 'horizontal':
        # --- 横向渲染逻辑 ---
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
             return char_boxes, img

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
            
            laid_out_lines.append({
                'text': line,
                'bbox': line_bbox,
                'width': line_width,
                'height': line_height,
                'y_top_offset': y_cursor - line_bbox[1]
            })
            
            y_cursor += line_height
            if i < len(lines) - 1:
                leading = max(1, int(line_height * 0.2))
                y_cursor += leading

        total_text_block_height = y_cursor

        # 防止溢出
        if total_text_block_height > box_height:
             print(f"Warning: Horizontal text block height ({total_text_block_height}) slightly exceeds box height ({box_height}).")
             total_text_block_height = box_height # Clamp for offset calculation

        offset_x = x_min + max(0, (box_width - max_line_width) // 2)
        offset_x = min(offset_x, x_max - max_line_width)
        
        offset_y = y_min + max(0, (box_height - total_text_block_height) // 2)
        offset_y = min(offset_y, y_max - total_text_block_height)
        if total_text_block_height > box_height:
             offset_y = y_min

        # 绘制横向文本
        for line_info in laid_out_lines:
            line_text = line_info['text']
            line_bbox = line_info['bbox']
            y_top_offset = line_info['y_top_offset']

            line_draw_x = offset_x - line_bbox[0] 
            line_draw_y = offset_y + y_top_offset 

            char_cursor_x_relative = 0
            for char in line_text:
                try:
                    char_bbox_raw = draw.textbbox((0, 0), char, font=font)
                except Exception:
                     char_bbox_raw = (0, 0, optimal_font_size * 0.6, optimal_font_size)

                char_draw_x = line_draw_x + char_cursor_x_relative
                char_draw_y = line_draw_y

                char_bbox_abs = (
                    char_draw_x + char_bbox_raw[0],
                    char_draw_y + char_bbox_raw[1],
                    char_draw_x + char_bbox_raw[2],
                    char_draw_y + char_bbox_raw[3]
                )

                char_boxes.append(char_bbox_abs)
                draw.rectangle(char_bbox_abs, outline="red", width=1)
                draw.text((char_draw_x, char_draw_y), char, font=font, fill="black")

                char_width = char_bbox_raw[2] - char_bbox_raw[0]
                char_cursor_x_relative += char_width

    elif direction == 'vertical':
        # --- 竖向渲染逻辑 ---
        columns = []
        current_column = ""
        for char in text:
            test_column = current_column + char
            # Quick check for column height feasibility
            temp_height = 0
            for c in test_column:
                 try:
                     cb = draw.textbbox((0,0), c, font=font)
                     ch = cb[3] - cb[1]
                 except:
                     ch = optimal_font_size
                 temp_height += ch + ch * 0.2
            if len(test_column) > 1: # Subtract leading after last char for check
                 temp_height -= ch * 0.2

            if temp_height <= box_height:
                current_column = test_column
            else:
                if current_column:
                    columns.append(current_column)
                    current_column = char
                else:
                    columns.append(char)
                    current_column = ""
        if current_column:
            columns.append(current_column)

        if not columns:
             draw.rectangle(box, outline="blue", width=2)
             return char_boxes, img

        # Layout columns
        laid_out_columns = []
        x_cursor = 0
        max_column_widths = []
        max_total_column_height = 0
        for i, col in enumerate(columns):
            col_chars_info = []
            col_width = 0
            col_height = 0
            for j, char in enumerate(col):
                try:
                    char_bbox = draw.textbbox((0, 0), char, font=font)
                except Exception:
                     char_bbox = (0, 0, optimal_font_size * 0.6, optimal_font_size)
                char_width = char_bbox[2] - char_bbox[0]
                char_height = char_bbox[3] - char_bbox[1]
                col_width = max(col_width, char_width)
                col_chars_info.append({
                    'char': char,
                    'bbox': char_bbox,
                    'width': char_width,
                    'height': char_height
                })
                col_height += char_height
                if j < len(col) - 1:
                    col_height += char_height * 0.2 # Leading

            max_column_widths.append(col_width)
            max_total_column_height = max(max_total_column_height, col_height)
            
            laid_out_columns.append({
                'chars': col_chars_info,
                'width': col_width,
                'height': col_height,
                'x_left_offset': x_cursor # x_cursor will be adjusted later for centering
            })
            
            x_cursor += col_width
            if i < len(columns) - 1:
                x_cursor += col_width * 0.1 # Column spacing

        total_text_block_width = x_cursor

        # 防止溢出
        if total_text_block_width > box_width:
             print(f"Warning: Vertical text block width ({total_text_block_width}) slightly exceeds box width ({box_width}).")
             total_text_block_width = box_width

        # Center the entire block of columns horizontally
        offset_x = x_min + max(0, (box_width - total_text_block_width) // 2)
        offset_x = min(offset_x, x_max - total_text_block_width)
        if total_text_block_width > box_width:
             offset_x = x_min

        # offset_y is for vertical centering within each column, handled per column below
        # But we need a base offset_y for the whole block vertically
        offset_y_block = y_min + max(0, (box_height - max_total_column_height) // 2)
        offset_y_block = min(offset_y_block, y_max - max_total_column_height)
        if max_total_column_height > box_height:
             offset_y_block = y_min


        # Draw vertical text
        current_column_x = offset_x
        for col_info in laid_out_columns:
            col_chars = col_info['chars']
            col_width = col_info['width']
            col_height = col_info['height']
            
            # Center this column's content vertically within the box
            # offset_y_block is the base, then adjust for this column's height
            offset_y_col = offset_y_block + max(0, (max_total_column_height - col_height) // 2)
            if col_height > box_height:
                 offset_y_col = y_min # Align tall column to top

            current_char_y = offset_y_col # Start y for the first char in this column

            # Center characters horizontally within this column's width
            # The column's left edge is at current_column_x
            # We want the character (considering its bbox) to be centered in col_width
            for char_info in col_chars:
                char = char_info['char']
                char_bbox_raw = char_info['bbox']
                char_height = char_info['height']

                # Calculate draw position
                # Horizontal centering within column width
                char_draw_x = current_column_x + (col_width - (char_bbox_raw[2] - char_bbox_raw[0])) / 2 - char_bbox_raw[0]
                # Vertical position
                char_draw_y = current_char_y - char_bbox_raw[1] # Align char's top to current_char_y

                char_bbox_abs = (
                    char_draw_x + char_bbox_raw[0],
                    char_draw_y + char_bbox_raw[1],
                    char_draw_x + char_bbox_raw[2],
                    char_draw_y + char_bbox_raw[3]
                )

                char_boxes.append(char_bbox_abs)
                draw.rectangle(char_bbox_abs, outline="red", width=1)
                draw.text((char_draw_x, char_draw_y), char, font=font, fill="black")

                # Move to next char position vertically
                current_char_y += char_height + char_height * 0.2 # Add leading

            # Move to next column position horizontally
            current_column_x += col_width + col_width * 0.1 # Add column spacing


    # 绘制文本框边界
    draw.rectangle(box, outline="blue", width=2)
    return char_boxes, img

def filter_valid_boxes(boxes, min_area=1):
    """
    过滤掉宽度或高度为0（或面积小于min_area）的边界框。

    Args:
        boxes (list of tuples): 输入的边界框列表，每个框是 (x1, y1, x2, y2)。
        min_area (int, optional): 最小有效面积。默认为1，即过滤掉线和点。

    Returns:
        list of tuples: 过滤后的有效边界框列表。
    """
    valid_boxes = []
    for box in boxes:
        if len(box) != 4:
            # 跳过格式不正确的框
            continue
            
        x1, y1, x2, y2 = box
        # 计算宽度和高度
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 检查面积是否大于等于最小面积
        if width * height >= min_area:
            valid_boxes.append(box)
        # else:
        #     print(f"过滤掉的框（面积为0或负数）: {box}")
            
    return valid_boxes

# --- 示例用法 ---
# 请确保将 'path/to/your/font.ttf' 替换为您系统上的有效字体文件路径
# 例如，在 Windows 上可能是 'C:/Windows/Fonts/arial.ttf' 或 'C:/Windows/Fonts/simsun.ttc'
# 在 Linux 上可能是 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' 或 '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
# 在 macOS 上可能是 '/System/Library/Fonts/Arial.ttf' 或 '/System/Library/Fonts/PingFang.ttc'

if __name__ == "__main__":
    # --- 横向渲染 ---
    char_boxes_h, img_h = render_text_in_box_with_visualization(
        image_width=800,
        image_height=600,
        text="这是一段比较长的测试文本，用来验证自适应字体大小和多行换行功能是否正常工作。Happy Coding!",
        box=(50, 50, 750, 550),
        font_path="/home/sxm/flux-workspace/Qwen-Image/util/simhei.ttf", # <--- 请替换为实际字体路径
        direction='horizontal'
    )
    if img_h:
        img_h.save("output_h.png")

    # --- 竖向渲染 ---
    char_boxes_v, img_v = render_text_in_box_with_visualization(
        image_width=800,
        image_height=600,
        text="这是一段比较长的测试文本，用来验证自适应字体大小和多行换行功能是否正常工作。Happy Coding!",
        box=(50, 50, 750, 550),
        font_path="/home/sxm/flux-workspace/Qwen-Image/util/simhei.ttf", # <--- 请替换为实际字体路径
        direction='vertical'
    )
    if img_v:
        img_v.save("output_v.png")



