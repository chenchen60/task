# src/failcases.py
import os
import cv2
from ultralytics import YOLO


MODEL_PATH = r"D:\lstm\ultralytics-main\nishuiceshi\weights\best.pt"

# 数据集目录（测试集图片）
IMG_DIR = r"D:\lstm\ultralytics-main\nishuiceshi\dataset_split\images\test"

# 结果保存目录
OUTPUT_DIR = r"D:\lstm\ultralytics-main\nishuiceshi\failcases"


os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

# 统计信息
fail_load = []
fail_detect = []
success = []

# 遍历图片
for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)

    # 尝试读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"[❌] 无法读取图像: {img_path}")
        fail_load.append(img_name)
        continue

    # 模型推理
    results = model(img, verbose=False)

    # 检查检测结果
    if len(results[0].boxes) == 0:
        print(f"[⚠️] 未检测到目标: {img_name}")
        fail_detect.append(img_name)

        # 保存失败案例图片
        save_path = os.path.join(OUTPUT_DIR, f"fail_{img_name}")
        cv2.imwrite(save_path, img)
    else:
        success.append(img_name)

print("\n====== 失败案例分析报告 ======")
print(f"总图片数: {len(os.listdir(IMG_DIR))}")
print(f"加载失败: {len(fail_load)} 张")
print(f"检测失败: {len(fail_detect)} 张")
print(f"检测成功: {len(success)} 张")

if fail_load:
    print("\n【加载失败的图片】")
    for f in fail_load:
        print("  -", f)

if fail_detect:
    print("\n【未检测到目标的图片】")
    for f in fail_detect:
        print("  -", f)

print(f"\n所有未检测到的图片已保存到: {OUTPUT_DIR}")
