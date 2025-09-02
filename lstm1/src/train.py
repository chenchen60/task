from ultralytics import YOLO
import os

def main():
    # 创建训练输出目录
    os.makedirs("runs/detect/train", exist_ok=True)

    # 加载 YOLOv8n 预训练模型
    model = YOLO("yolov8n.pt")  # 可换成 yolov8s/yolov8m

    # 训练模型
    model.train(
        data="D:\\lstm1\\dataset_split\\dataset.yaml",    # 数据集配置文件
        epochs=50,              # 训练轮次
        imgsz=640,              # 输入图像大小
        batch=16,               # batch size
        device="CPU",                # CPU 或 GPU ("0")
        name="swimming_pool",    # 保存结果文件夹名
        pretrained=True,         # 使用预训练权重
        augment=True,            # 数据增强
        plots=True               # 绘制训练曲线
    )

if __name__ == "__main__":
    main()
