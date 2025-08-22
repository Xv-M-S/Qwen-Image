# remove_detach_lines.py

def remove_detach_lines(input_file, output_file):
    """
    读取 input_file，删除包含 'detach' 的行，保存到 output_file
    """
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    # 过滤掉包含 'detach' 的行（忽略大小写）
    filtered_lines = []
    for line in lines:
        if 'detach' not in line.lower():
            filtered_lines.append(line)
        else:
            count += 1
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(filtered_lines)
    
    print(f"✅ 已删除 {count} 行包含 'detach' 的内容")
    print(f"✅ 已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    input_path = "reclc.txt"          # 修改为你的文件路径
    output_path = "reclc_clean.txt"   # 输出文件名
    remove_detach_lines(input_path, output_path)