import cv2
import torch
import sys
import numpy as np
import warnings
from pathlib import Path
import time

# --- 0. 屏蔽烦人的警告 (pkg_resources) ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- ★★★ 核心修复：针对 PyTorch 2.6+ 的补丁 ★★★ ---
_original_torch_load = torch.load


def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


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


class YOLO_Detector:
    def __init__(self, weights, device='', conf_thresh=0.5, iou_thresh=0.45, imgsz=640):
        self.device = select_device(device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        print(f"正在加载 PyTorch 模型: {weights} ...")
        try:
            self.model = attempt_load(weights, map_location=self.device)

            for m in self.model.modules():
                t = type(m)
                if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                    m.recompute_scale_factor = None

            self.stride = int(self.model.stride.max())
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

            self.model.eval()
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).float() / 255.0)

            print("模型加载成功 (PyTorch Mode)")

        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def detect(self, image_raw):
        """
        执行推理
        :return: [x, y, w, h, conf] 或者 None
        """
        # 1. 预处理
        img, ratio, pad = self.preprocess_image(image_raw)

        # 2. 推理
        with torch.no_grad():
            pred = self.model(img)[0]

        # 3. NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=None, max_det=1000)

        det = pred[0]

        if len(det):
            # 缩放坐标回原图
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_raw.shape).round()

            # 寻找最高置信度的目标
            best_det = det[det[:, 4].argmax()]

            x1, y1, x2, y2 = map(int, best_det[:4])
            w = x2 - x1
            h = y2 - y1

            # ★★★ 修改点 1：获取置信度 conf ★★★
            conf = float(best_det[4])

            # ★★★ 修改点 2：返回列表增加 conf ★★★
            return [x1, y1, w, h, conf]

        return None

    def preprocess_image(self, img0):
        img, ratio, pad = letterbox(img0, self.imgsz, stride=self.stride)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
        return img, ratio, pad

    def destroy(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    # ================= 配置 =================
    weights_path = "../model/TensorRT_10/yolov5s_GLAD.pt"
    video_path = "/home/verser/Videos/phantom13.mp4"
    # =======================================

    detector = YOLO_Detector(weights_path, device='', conf_thresh=0.5)

    cv2.namedWindow("PyTorch YOLO Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PyTorch YOLO Detector", 640, 640)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频源 {video_path}")
        return

    print("✅ 开始推理。按 'q' 键退出...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频结束。")
                break

            t_start = time.time()

            # 执行检测
            box = detector.detect(frame)

            fps = 1.0 / (time.time() - t_start) if (time.time() - t_start) > 0 else 0

            # 绘制结果
            if box is not None:
                # ★★★ 修改点 3：这里解包出 5 个变量，包含 conf ★★★
                x, y, w, h, conf = box

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # ★★★ 修改点 4：将 conf 格式化为字符串显示 ★★★
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS 固定显示在左上角
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("PyTorch YOLO Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n用户停止。")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.destroy()


if __name__ == "__main__":
    main()