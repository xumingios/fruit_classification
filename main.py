'''
启动：
python .\infer.py --model_dir=yolov3_mobilenet_v3_large_voc --camera_id=0 --use_gpu=True
'''

import os
import time
import argparse
import numpy as np
import cv2
import paddle
from PIL import Image
from preprocess import preprocess, Resize, NormalizeImage, Permute

class FruitDetectionSystem:
    def __init__(self, model_dir, use_gpu=False, threshold=0.5):
        """
        初始化水果检测系统
        :param model_dir: 模型目录路径
        :param use_gpu: 是否使用GPU
        :param threshold: 检测阈值
        """
        self.threshold = threshold
        self.labels = self.load_labels(model_dir)
        self.predictor = self.load_predictor(model_dir, use_gpu)
        self.preprocess_ops = self.build_preprocess_ops()

    def load_labels(self, model_dir):
        """从配置文件加载类别标签"""
        cfg_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        return cfg['label_list']

    def build_preprocess_ops(self):
        """构建预处理流水线"""
        return [
            Resize(target_size=608, keep_ratio=False),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Permute()
        ]

    def load_predictor(self, model_dir, use_gpu):
        """加载预测模型"""
        config = paddle.inference.Config(
            os.path.join(model_dir, 'model.pdmodel'),
            os.path.join(model_dir, 'model.pdiparams')
        )
        if use_gpu:
            config.enable_use_gpu(200, 0)
        else:
            config.disable_gpu()
        return paddle.inference.create_predictor(config)

    def preprocess_frame(self, frame):
        """预处理摄像头帧"""
        im_info = {
            'scale_factor': np.array([1., 1.], dtype=np.float32),
            'im_shape': None,
            'input_shape': [3, 608, 608]  # 与模型输入尺寸匹配
        }
        # 转换颜色空间并预处理
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for op in self.preprocess_ops:
            im, im_info = op(im, im_info)
        return np.expand_dims(im, axis=0)  # 添加batch维度

    def detect(self, frame):
        """执行检测"""
        # 预处理
        input_data = self.preprocess_frame(frame)
        
        # 设置模型输入
        input_handle = self.predictor.get_input_handle('image')
        input_handle.copy_from_cpu(input_data)
        
        # 执行预测
        self.predictor.run()
        
        # 获取输出
        output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        np_boxes = output_handle.copy_to_cpu()
        
        # 过滤低置信度结果
        keep = np_boxes[:, 1] > self.threshold
        return {'boxes': np_boxes[keep]}

    def visualize(self, frame, results):
        """可视化检测结果"""
        # 转换为PIL格式
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 绘制检测框
        if 'boxes' in results:
            draw = ImageDraw.Draw(pil_img)
            color_list = self.get_color_map_list(len(self.labels))
            
            for box in results['boxes']:
                clsid, score, xmin, ymin, xmax, ymax = map(float, box)
                clsid = int(clsid)
                
                # 绘制边界框
                color = tuple(color_list[clsid])
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                
                # 绘制标签
                text = f"{self.labels[clsid]} {score:.2f}"
                font = ImageFont.load_default()
                text_w, text_h = font.getsize(text)
                draw.rectangle([xmin, ymin-text_h, xmin+text_w, ymin], fill=color)
                draw.text((xmin, ymin-text_h), text, fill=(255,255,255))
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def run_camera(self, camera_id=0):
        """启动摄像头实时检测"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return

        print("实时水果检测中 (按Q退出)...")
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            results = self.detect(frame)
            
            # 可视化
            vis_frame = self.visualize(frame, results)
            
            # 计算并显示FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Fruit Detection', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, 
                       help="包含model.pdmodel和model.pdiparams的目录")
    parser.add_argument("--camera_id", type=int, default=0, 
                       help="摄像头设备ID")
    parser.add_argument("--use_gpu", type=bool, default=False,
                       help="是否使用GPU加速")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="检测置信度阈值")
    args = parser.parse_args()

    # 创建并运行系统
    detector = FruitDetectionSystem(
        model_dir=args.model_dir,
        use_gpu=args.use_gpu,
        threshold=args.threshold
    )
    detector.run_camera(args.camera_id)