# compare_files_ignore_dollar.py

import re

def clean_line(line):
    """
    删除行中所有的 $数字，例如：
    '$123: tensor' -> ': tensor'
    'x = add($1, $2)' -> 'x = add(, )'
    注意：我们保留空格结构，便于比对
    """
    return re.sub(r'\$\d+', '', line)

def compare_files_ignore_dollar(file1, file2):
    """
    逐行比对两个文件，忽略 $数字，找出第一个不同的行
    """
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return

    min_len = min(len(lines1), len(lines2))
    
    for i in range(min_len):
        clean1 = clean_line(lines1[i].strip())
        clean2 = clean_line(lines2[i].strip())
        
        if clean1 != clean2:
            print(f"❌ 发现第一个不同行 (行号 {i+1}):")
            print(f"  {file1}: '{lines1[i].strip()}'")
            print(f"  {file2}: '{lines2[i].strip()}'")
            print(f"  清理后:")
            print(f"    {clean1}")
            print(f"    {clean2}")
            return
    
    # 如果前面都相同，检查长度
    if len(lines1) != len(lines2):
        print(f"✅ 前 {min_len} 行相同，但文件长度不同：")
        print(f"  {file1} 有 {len(lines1)} 行")
        print(f"  {file2} 有 {len(lines2)} 行")
        if len(lines1) > len(lines2):
            print(f"  多出行 {min_len+1} from {file1}: '{lines1[min_len].strip()}'")
        else:
            print(f"  多出行 {min_len+1} from {file2}: '{lines2[min_len].strip()}'")
        return

    print("✅ 两个文件在忽略 $数字 后完全相同！")

# 使用示例
if __name__ == "__main__":
    file1 = "reclc_clean.txt"      # 修改为你的第一个文件
    file2 = "ori_clc.txt" # 修改为你的第二个文件
    compare_files_ignore_dollar(file1, file2)