# 基于yolo的实时水果检测

### 一、介绍
基于paddlepaddle框架，使用yolov3目标检测算法实现实时水果检测（苹果、橘子、香蕉）。

### 二、目录
|-output 存放检测好的视频

|-yolov3_mobilenet_v3_large_voc 模型权重文件

|-preprocess.py 图片预处理

|-visualize.py 可视化操作

### 三、建议环境
python=3.10

### 四、快速使用
1、安装paddle2.0.0以上版本

`pip install paddlepaddle`

2、安装依赖库

`pip install -r requirements.txt`

3、运行，检测视频结果在output文件夹中

`python .\infer.py --model_dir=yolov3_mobilenet_v3_large_voc --camera_id=0 --use_gpu=True`
