import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument("image_path", type=str, help="输入图片路径")
    parser.add_argument("--save", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--save_dir", type=str, default="runs/detect/infer", help="保存结果的目录")
    args = parser.parse_args()

    # 加载模型
    model = YOLO('runs/detect/swimming_pool2/weights/best.pt')

    # 推理
    results = model(args.image_path)

    # 输出类别与置信度
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        cls_name = model.names[cls_id]
        print(f"类别: {cls_name}, 置信度: {conf:.3f}")

    # 可选保存可视化结果
    if args.save:
        save_path = f"{args.save_dir}/result.jpg"
        results[0].plot(save=True, filename=save_path)
        print(f"结果已保存到: {save_path}")

if __name__ == "__main__":
    main()
