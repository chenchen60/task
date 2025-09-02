# split_dataset_with_analysis.py
import os
import random
import shutil
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib

# -------------------------------
# 中文显示设置，解决字体缺失问题
# -------------------------------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def analyze_dataset(images_dir, labels_dir):
    """
    数据理解分析，并将类别2自动改为1
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # 递归查找所有 jpg/jpeg/png 文件，过滤 macOS 生成的 ._ 文件
    image_files = [f for f in images_dir.rglob("*.*") 
                   if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                   and not f.name.startswith("._")]

    total_images = len(image_files)
    sizes = []
    class_counts = Counter()
    corrupted_images = []
    missing_labels = []

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue
        try:
            # 检查图片是否损坏
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                sizes.append(im.size)
            
            # 读取标签文件，类别2改为1
            new_lines = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    if class_id == 2:
                        class_id = 1
                    class_counts[class_id] += 1
                    parts[0] = str(class_id)
                    new_lines.append(" ".join(parts))
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")

        except Exception as e:
            corrupted_images.append(img_path.name)
            print(f"[❌] 图片损坏: {img_path} | 错误: {e}")

    print(f"总图片数: {total_images}")
    print(f"损坏图片数: {len(corrupted_images)}")
    print(f"缺失标签图片数: {len(missing_labels)}")
    print(f"类别统计: {class_counts}")

    # 可视化类别分布
    plt.figure(figsize=(5,4))
    plt.bar([str(k) for k in class_counts.keys()], class_counts.values())
    plt.xlabel("类别")
    plt.ylabel("标注数量")
    plt.title("类别分布")
    plt.savefig("class_distribution.png")
    plt.close()

    # 可视化图片尺寸分布
    if sizes:
        widths, heights = zip(*sizes)
        plt.figure(figsize=(6,4))
        plt.scatter(widths, heights, alpha=0.5)
        plt.xlabel("宽度")
        plt.ylabel("高度")
        plt.title("图片尺寸分布")
        plt.savefig("image_size_distribution.png")
        plt.close()

    return corrupted_images, missing_labels, image_files

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    划分数据集，同时处理类别2为1
    """
    random.seed(seed)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # 创建输出文件夹
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 获取所有图片，过滤 macOS ._ 文件
    image_files = [f for f in images_dir.rglob("*.*") 
                   if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                   and not f.name.startswith("._")]
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    print(f"训练集: {len(train_files)} | 验证集: {len(val_files)} | 测试集: {len(test_files)}")

    def check_and_copy(img_path, label_path, dst_img, dst_label):
        try:
            with Image.open(img_path) as im:
                im.verify()
            shutil.copy(img_path, dst_img)
            # 再次修改类别2为1
            new_lines = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    if class_id == 2:
                        class_id = 1
                    parts[0] = str(class_id)
                    new_lines.append(" ".join(parts))
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
        except Exception as e:
            print(f"[❌] 图片或标签损坏，跳过: {img_path} | 错误: {e}")

    def copy_files(file_list, split):
        for img_path in file_list:
            label_path = labels_dir / (img_path.stem + ".txt")
            dst_img = output_dir / "images" / split / img_path.name
            dst_label = output_dir / "labels" / split / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"[⚠️] 没找到标签: {label_path}")
                continue
            check_and_copy(img_path, label_path, dst_img, dst_label)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    # 生成 dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        f.write("  0: normal\n")
        f.write("  1: drowning\n")

    print(f"数据划分完成，配置文件已生成: {yaml_path}")

if __name__ == "__main__":
    images_dir = "test/images"
    labels_dir = "test/labels"
    output_dir = "dataset_split"

    # 数据分析并修改类别2
    corrupted_images, missing_labels, image_files = analyze_dataset(images_dir, labels_dir)
    if corrupted_images:
        print("损坏图片列表:", corrupted_images)
    if missing_labels:
        print("缺失标签列表:", missing_labels)

    print(f"总图片数（分析结果）: {len(image_files)}")

    # 划分数据集
    split_dataset(images_dir, labels_dir, output_dir)
