import cv2
import torch
import sys
import time
import numpy as np
import warnings  # 新增：用于忽略警告
from pathlib import Path

# --- 0. 屏蔽烦人的警告 (pkg_resources) ---
warnings.filterwarnings("ignore", category=UserWarning)


_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    # 如果调用时没有指定 weights_only，则强制设为 False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 应用补丁
torch.load = safe_torch_load
# ---------------------------------------------------

# --- 1. 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
YOLOv5_ROOT = ROOT / '../yolov5'

if str(YOLOv5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOv5_ROOT))

# 导入 YOLOv5 模块
try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
except ImportError as e:
    print(f"YOLOv5 导入错误: {e}")
    print("请确保 yolov5 文件夹在当前目录下，且安装了 requirements.txt")
    sys.exit(1)

def run_legacy_inference(weights, source=0, conf_thres=0.5, device=''):

    device = select_device(device)

    print(f"正在使用 attempt_load 加载模型: {weights} ...")
    try:
        # 此时调用的 torch.load 已经被上面的补丁修改过，可以正常加载了
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
        # 打印更详细的错误堆栈，方便调试
        import traceback
        traceback.print_exc()
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
                # 兼容性处理：如果 scale_coords 报错，可能需要改用 scale_boxes
                # 新版 YOLOv5 utils.general 可能用 scale_boxes 替代了 scale_coords
                try:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame0.shape).round()
                except NameError:
                    from utils.general import scale_boxes
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'

                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(frame0, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame0, label, (p1[0], p1[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ==========================================
        # ★★★ 计算并显示 FPS ★★★
        # ==========================================
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

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
    # 请确保路径正确
    my_model = "../model/TensorRT_10/yolov5s_GLAD.pt"
    VIDEO_PATH = "/home/verser/Videos/phantom13.mp4"

    run_legacy_inference(my_model, source=VIDEO_PATH)