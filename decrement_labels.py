#!/usr/bin/env python3
"""
YOLO标签类别修改脚本

功能：将指定目录下train和val子目录中所有标签文件的类别标签加1或减1
用法：python decrement_labels.py <labels_directory> [--increment]

例如：
- 减1: python decrement_labels.py /home/yu/zjh/SSDA-YOLO-copy/dataset3/yolo_K2C_format_2/labels
- 加1: ss
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime


def create_backup(labels_dir):
    """
    创建标签目录的备份
    
    Args:
        labels_dir (str): 标签目录路径
    
    Returns:
        str: 备份目录路径，失败时返回None
    """
    try:
        labels_path = Path(labels_dir)
        parent_dir = labels_path.parent
        
        # 创建带时间戳的备份目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{labels_path.name}_backup_{timestamp}"
        backup_path = parent_dir / backup_name
        
        print(f"创建备份: {backup_path}")
        
        # 复制整个目录
        shutil.copytree(labels_dir, backup_path)
        
        print(f"✓ 备份创建成功: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        print(f"× 创建备份失败: {e}")
        return None


def modify_label_file(file_path, increment=False):
    """
    处理单个标签文件，将类别标签加1或减1
    
    Args:
        file_path (str): 标签文件路径
        increment (bool): True为加1，False为减1
    
    Returns:
        bool: 是否成功处理
    """
    operation = "加1" if increment else "减1"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                new_lines.append('\n')
                continue
                
            parts = line.split()
            if len(parts) >= 5:  # YOLO格式至少5列：class x y w h
                try:
                    class_id = int(parts[0])
                    new_class_id = class_id + 1 if increment else class_id - 1
                    
                    # 确保类别不会变成负数
                    if new_class_id < 0:
                        print(f"警告: {file_path} 中类别 {class_id} {operation}后为负数，跳过此行")
                        new_lines.append(line + '\n')
                        continue
                    
                    # 替换类别标签
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts) + '\n')
                    modified = True
                    
                except ValueError:
                    print(f"警告: {file_path} 中第一列不是有效的整数，跳过此行: {line}")
                    new_lines.append(line + '\n')
            else:
                print(f"警告: {file_path} 中格式不正确，跳过此行: {line}")
                new_lines.append(line + '\n')
        
        # 只有在修改了内容时才写回文件
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True
        else:
            print(f"文件 {file_path} 没有需要修改的内容")
            return False
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def process_directory(labels_dir, create_backup_flag=False, increment=False):
    """
    处理指定目录下的train和val子目录
    
    Args:
        labels_dir (str): 标签目录路径
        create_backup_flag (bool): 是否创建备份
        increment (bool): True为加1，False为减1
    """
    labels_path = Path(labels_dir)
    operation = "加1" if increment else "减1"
    
    if not labels_path.exists():
        print(f"错误: 目录 {labels_dir} 不存在")
        return
    
    if not labels_path.is_dir():
        print(f"错误: {labels_dir} 不是一个目录")
        return
    
    # 创建备份
    backup_path = None
    if create_backup_flag:
        backup_path = create_backup(labels_dir)
        if backup_path is None:
            print("备份创建失败，是否继续处理？")
            confirm = input("继续？(y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("操作已取消")
                return
        else:
            print(f"备份已保存到: {backup_path}")
    
    # 检查train和val子目录
    subdirs = ['train', 'val']
    total_files = 0
    processed_files = 0
    
    for subdir in subdirs:
        subdir_path = labels_path / subdir
        
        if not subdir_path.exists():
            print(f"警告: 子目录 {subdir_path} 不存在，跳过")
            continue
            
        if not subdir_path.is_dir():
            print(f"警告: {subdir_path} 不是一个目录，跳过")
            continue
        
        print(f"\n处理目录: {subdir_path}")
        
        # 遍历所有.txt文件
        txt_files = list(subdir_path.glob('*.txt'))
        
        if not txt_files:
            print(f"在 {subdir_path} 中没有找到.txt文件")
            continue
        
        print(f"找到 {len(txt_files)} 个标签文件")
        
        for txt_file in txt_files:
            total_files += 1
            print(f"处理文件: {txt_file.name}")
            
            if modify_label_file(str(txt_file), increment):
                processed_files += 1
                print(f"✓ 成功处理: {txt_file.name}")
            else:
                print(f"× 跳过文件: {txt_file.name}")
    
    print(f"\n=== 处理完成 ===")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files}")
    print(f"跳过文件: {total_files - processed_files}")
    print(f"操作类型: 类别标签{operation}")
    
    if backup_path:
        print(f"原始文件备份位置: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将YOLO标签文件中的类别标签加1或减1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 类别标签减1（默认）
  python decrement_labels.py /path/to/labels
  
  # 类别标签加1
  python decrement_labels.py /path/to/labels --increment
  
  # 创建备份后处理（减1）
  python decrement_labels.py -b /path/to/labels
  
  # 创建备份后处理（加1）
  python decrement_labels.py -b --increment /home/yu/zjh/SSDA-YOLO-copy/dataset3/yolo_K2C_format_2/labels
        """
    )
    
    parser.add_argument('labels_dir', 
                       help='标签目录路径（包含train和val子目录）')
    
    parser.add_argument('--backup', '-b',
                       action='store_true',
                       help='处理前创建备份')
    
    parser.add_argument('--increment', '-i',
                       action='store_true',
                       help='将类别标签加1（默认是减1）')
    
    args = parser.parse_args()
    
    operation = "加1" if args.increment else "减1"
    
    # 确认操作
    print(f"即将处理目录: {args.labels_dir}")
    print(f"这将修改train和val子目录中所有.txt文件的类别标签（{operation}）")
    
    if args.backup:
        print("✓ 将在处理前创建备份")
    else:
        print("⚠ 不会创建备份")
    
    confirm = input("确认继续？(y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("操作已取消")
        return
    
    process_directory(args.labels_dir, args.backup, args.increment)


if __name__ == '__main__':
    main()