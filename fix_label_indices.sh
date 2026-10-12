#!/bin/bash
# 修复 YOLO 标签文件中的类别索引偏移问题：将所有类别索引从 1 改为 0
# 针对 dataset3 下四个子数据集的标签目录

set -euo pipefail

BASE="/home/yu/zjh/SSDA-YOLO-copy/dataset3"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
BACKUP_DIR="$BASE/labels_backup_$TIMESTAMP"

echo "=========================================="
echo "修复 YOLO 标签类别索引偏移 (1->0)"
echo "=========================================="
echo "基础目录: $BASE"
echo "备份目录: $BACKUP_DIR"
echo

# 1. 创建备份目录
echo "1. 创建备份..."
mkdir -p "$BACKUP_DIR"

# 2. 备份所有标签目录
echo "2. 备份原始标签..."
for sub in yolo_kitti_format_2 yolo_K2C_format_2 yolo_C2K_format_2 yolov5_city_format_2; do
    for split in train val; do
        src="$BASE/$sub/labels/$split"
        if [ -d "$src" ]; then
            dst="$BACKUP_DIR/$sub/labels"
            mkdir -p "$dst"
            cp -a "$src" "$dst/"
            echo "✓ 备份: $src -> $dst/"
        else
            echo "✗ 跳过 (目录不存在): $src"
        fi
    done
done

echo
echo "3. 开始修复标签索引..."

# 3. 修复标签文件：将首列类别索引从 1 改为 0
total_files=0
fixed_files=0

for sub in yolo_kitti_format_2 yolo_K2C_format_2 yolo_C2K_format_2 yolov5_city_format_2; do
    for split in train val; do
        label_dir="$BASE/$sub/labels/$split"
        if [ -d "$label_dir" ]; then
            echo "处理目录: $label_dir"
            
            # 查找所有 .txt 标签文件
            while IFS= read -r -d '' file; do
                ((total_files++))
                
                # 检查文件是否包含以 "1 " 开头的行（需要修复的情况）
                if grep -q "^1 " "$file" 2>/dev/null; then
                    # 使用 sed 将所有行首的 "1 " 替换为 "0 "
                    sed -i 's/^1 /0 /' "$file"
                    ((fixed_files++))
                    echo "  ✓ 修复: $(basename "$file")"
                fi
            done < <(find "$label_dir" -name "*.txt" -type f -print0)
        fi
    done
done

echo
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo "处理文件总数: $total_files"
echo "实际修复文件: $fixed_files"
echo "备份位置: $BACKUP_DIR"
echo
echo "如需恢复原始标签，请运行："
echo "  rm -rf $BASE/{yolo_kitti_format_2,yolo_K2C_format_2,yolo_C2K_format_2,yolov5_city_format_2}/labels"
echo "  cp -a $BACKUP_DIR/* $BASE/"
echo

# 4. 验证修复结果（抽样检查）
echo "4. 验证修复结果（抽样）..."
for sub in yolo_kitti_format_2 yolo_K2C_format_2 yolo_C2K_format_2 yolov5_city_format_2; do
    for split in train val; do
        label_dir="$BASE/$sub/labels/$split"
        if [ -d "$label_dir" ]; then
            sample_file=$(find "$label_dir" -name "*.txt" -type f | head -n 1)
            if [ -n "$sample_file" ]; then
                echo "=== 样本检查: $sample_file"
                head -n 3 "$sample_file" | sed 's/^/    /'
            fi
        fi
    done
done

echo
echo "修复脚本执行完毕。现在所有类别索引应为 0，匹配 YAML 配置 nc=1, names=['car']"