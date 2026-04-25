import cv2
import torch
import sys
import time
import numpy as np
from pathlib import Path

# --- 1. 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
YOLOv5_ROOT = ROOT / 'yolov5'

if str(YOLOv5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOv5_ROOT))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device

def run_legacy_inference(weights, source=0, conf_thres=0.5, device=''):

    device = select_device(device)

    print(f"正在使用 attempt_load 加载模型: {weights} ...")
    try:
        model = attempt_load(weights, map_location=device)

        for m in model.modules():
            t = type(m)
            if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # 强制补上这个属性
        # ==========================================

        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names
        imgsz = 640
        model.eval()

    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开摄像头/视频: {source}")
        return

    print("开始推理 (Legacy Mode)...")

    # --- 初始化 FPS 计数器 ---
    prev_time = 0


    cv2.namedWindow("YOLOv5 Legacy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv5 Legacy", 640, 640)

    while True:
        ret, frame0 = cap.read()
        if not ret: break

        # --- 预处理 ---
        img = letterbox(frame0, imgsz, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]

        # --- 推理 ---
        with torch.no_grad():
            pred = model(img)[0]

        # --- NMS ---
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, max_det=1000)

        # --- 画图 ---
        for i, det in enumerate(pred):
            if len(det):
                # 如果运行报错 scale_coords 不存在，请改回 scale_boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'

                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(frame0, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame0, label, (p1[0], p1[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ==========================================
        # ★★★ 新增：计算并显示 FPS ★★★
        # ==========================================
        curr_time = time.time()
        # 防止除以零
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # 在左上角显示 FPS (红色字体)
        cv2.putText(frame0, f"FPS: {fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # ==========================================

        cv2.imshow("YOLOv5 Legacy", frame0)

        # 按 'q' 退出
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    my_model = "./model/TensorRT_10/yolov5s_GLAD.pt"
    # 这里使用了你指定的视频路径
    VIDEO_PATH = "/home/verse/Videos/phantom13.mp4"

    run_legacy_inference(my_model, source=VIDEO_PATH)