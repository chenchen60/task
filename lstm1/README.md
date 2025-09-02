# 所需环境

python 3.8

安装依赖：pip install -r requirements.txt

# 电力预测模块

用电量趋势图：python dianliyuce/moxinxuanzetu.py

确保电力数据文件：project/dianliyuce/dianli.xlsx

运行LSTM模型预测：python dianliyuce/lstm.py

# YOLOv8溺水检测模块

## 数据准备

1）数据放在同文件的根目录下

2）划分数据集：python dataset_split/split_dataset.py --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1

## 模型训练

2）python src/train.py --data dataset_split/dataset.yaml --weights yolov8n.pt --epochs 50 --batch 16 --img 640

## 模型推理

python src/infer.py 图像路径  --save --save_dir runs/detect/



