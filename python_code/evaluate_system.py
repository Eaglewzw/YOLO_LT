import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import glob
import sys

# ================= 1. 导入你的系统模块 =================
from LightTrack_FastTrack import LightTrackEngine

# 尝试导入 YOLO (兼容 TRT10 和 TRT8)
try:
    from modules_trt10.YOLO_Engine_TRT10 import YOLO_Detector

    USE_TRT10 = True
    print("Detected: TensorRT 10 Module")
except ImportError:
    try:
        from YOLO_Engine import YOLO_Detector

        USE_TRT10 = False
        print("Detected: TensorRT 8.6 Module")
    except ImportError:
        print("❌ 错误: 找不到 YOLO 引擎模块，请检查文件名")
        sys.exit(1)

# 尝试导入 MOD
try:
    from MOD2 import MOD2_global
except ImportError:
    print("⚠️ 警告: 找不到 MOD2 模块，将仅使用 YOLO+LightTrack")


    def MOD2_global(prev, curr):
        return None


# ================= 2. 定义系统包装类 =================
class DroneTrackerSystem:
    def __init__(self):
        # --- 配置 ---
        self.VISUAL_FAIL_THRESHOLD = 30
        self.MOD_FAIL_THRESHOLD = 60

        # --- 模型路径 (请根据实际情况修改) ---
        self.init_model = "./model/ligthtrack_init.pt"
        self.update_model = "./model/ligthtrack_update.pt"

        if USE_TRT10:
            self.yolo_path = "./model/TensorRT_10/yolov5s_GLAD.pt"
        else:
            self.yolo_path = "./model/TensorRT_8/yolov5s.engine"  # 修改为你的 TRT8 引擎路径

        # --- 初始化 ---
        if not os.path.exists(self.init_model):
            print(f"❌ 模型缺失: {self.init_model}")
            sys.exit(1)

        self.tracker = LightTrackEngine(self.init_model, self.update_model, device='cuda')
        self.yolo = YOLO_Detector(self.yolo_path)

        self.reset_state()

    def reset_state(self):
        """处理新视频前重置内部变量"""
        self.tracking_state = False
        self.visual_fail_count = 0
        self.mod_fail_count = 0
        self.prev_frame = None

    def process_frame(self, curr_frame):
        """
        输入: 当前帧图像
        输出: [x1, y1, x2, y2, score] (左上右下格式) 或 None
        """
        bbox = None
        score = 0.0

        # --- A. 搜索模式 (Searching) ---
        if not self.tracking_state:
            # === 1. 尝试 YOLO 检测 ===
            if self.visual_fail_count < self.VISUAL_FAIL_THRESHOLD:
                det = self.yolo.detect(curr_frame)

                # 不仅检查 is not None，还要检查长度是否足够
                if det is not None and len(det) >= 4:
                    # 只有当检测框合法时才进入
                    bbox = det[:4]
                    score = det[4] if len(det) > 4 else 1.0
                    self.visual_fail_count = 0

                    # 初始化跟踪器
                    self.tracker.init(curr_frame, bbox)
                    self.tracking_state = True
                else:
                    # YOLO 没检测到东西，或者返回了空列表 []
                    self.visual_fail_count += 1

            # === 2. 尝试 MOD 运动检测 ===
            # (仅当 YOLO 失败次数过多，或 YOLO 刚失败时进入，取决于你的策略，这里保持原逻辑)
            elif self.prev_frame is not None:
                mod_box = MOD2_global(self.prev_frame, curr_frame)

                # 【关键修改】同样检查 MOD 返回值的长度
                if mod_box is not None and len(mod_box) >= 4:
                    bbox = mod_box
                    score = 0.5
                    self.tracker.init(curr_frame, bbox)
                    self.tracking_state = True
                    self.mod_fail_count = 0
                else:
                    self.mod_fail_count += 1
                    if self.mod_fail_count > self.MOD_FAIL_THRESHOLD:
                        self.visual_fail_count = 0

                        # --- B. 跟踪模式 (Tracking) ---
        else:
            bbox, score = self.tracker.track(curr_frame)

            # 检查跟踪结果是否有效
            if bbox is not None and len(bbox) >= 4:
                # 跟踪丢失判断 (阈值建议尝试 0.7 - 0.9)
                if score <= 0.98:
                    self.tracking_state = False
                    self.visual_fail_count = 0
                    bbox = None
            else:
                # 如果 track 返回了异常空值
                self.tracking_state = False
                bbox = None

        self.prev_frame = curr_frame.copy()

        # --- C. 结果格式化 ---
        if bbox is not None and len(bbox) >= 4:
            # 转换为 [x1, y1, x2, y2, score]
            try:
                x, y, w, h = map(float, bbox[:4])
                return [x, y, x + w, y + h, score]
            except ValueError:
                # 防止 bbox 内部包含非数字导致的崩溃
                return None

        return None


# ================= 3. 数据解析工具 =================
def parse_xml_gt(xml_file):
    """解析 PASCAL VOC 格式 XML"""
    if not os.path.exists(xml_file):
        return []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        gt_boxes = []

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            gt_boxes.append([xmin, ymin, xmax, ymax])
        return gt_boxes
    except Exception as e:
        print(f"XML Error: {xml_file} - {e}")
        return []


def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6

    return inter / union


# ================= 4. 评估核心类 =================
class Evaluator:
    def __init__(self):
        self.all_preds = []
        self.all_gts = []
        self.total_gt_objects = 0

    def add_batch(self, preds, gts):
        self.all_preds.append(preds)
        self.all_gts.append(gts)
        self.total_gt_objects += len(gts)

    def compute_metrics(self, iou_thresh=0.5):
        tp_list = []
        conf_list = []
        num_gt = self.total_gt_objects

        for preds, gts in zip(self.all_preds, self.all_gts):
            # 这一帧没有预测出目标
            if preds is None:
                continue

            pred_box = preds[:4]
            score = preds[4]
            conf_list.append(score)

            # 这一帧有预测，但没有 GT (误检)
            if len(gts) == 0:
                tp_list.append(0)
                continue

            # 找最大 IoU
            best_iou = 0
            for gt in gts:
                iou = compute_iou(pred_box, gt)
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= iou_thresh:
                tp_list.append(1)
            else:
                tp_list.append(0)

        if len(tp_list) == 0:
            return 0.0, 0.0, 0.0

        # 排序
        tp_list = np.array(tp_list)
        conf_list = np.array(conf_list)
        sort_idx = np.argsort(conf_list)[::-1]
        tp_list = tp_list[sort_idx]

        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(1 - tp_list)

        recalls = tp_cumsum / (num_gt + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # 使用 11点插值法 (VOC2007) 或 积分法 计算 AP
        ap = np.trapz(precisions, recalls)

        return precisions[-1], recalls[-1], ap


# ================= 5. 主程序 (路径逻辑已修改) =================
def main():
    # --- 配置你的数据集根目录 ---
    # 根据你的描述，根目录是 ARD-MAV
    DATASET_ROOT = "/media/verser/robot/Dataset/ARD-MAV"  # 如果代码不在 ARD-MAV 旁边，请改为绝对路径，例如 "/home/user/data/ARD-MAV"

    VIDEO_DIR = os.path.join(DATASET_ROOT, "videos")
    ANNOTATION_DIR = os.path.join(DATASET_ROOT, "Annotations")

    # 检查路径是否存在
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 错误: 找不到视频目录 {VIDEO_DIR}")
        return

    # 获取所有 .mp4 文件
    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if len(video_files) == 0:
        print(f"❌ 错误: 在 {VIDEO_DIR} 下没有找到 .mp4 文件")
        return

    print(f"✅ 找到 {len(video_files)} 个视频文件")

    system = DroneTrackerSystem()
    evaluator = Evaluator()

    # --- 遍历视频 ---
    for v_path in video_files:
        # 获取文件名 (不带后缀) 例如: "phantom39"
        video_name = os.path.splitext(os.path.basename(v_path))[0]

        # 构建该视频对应的 XML 目录: ARD-MAV/Annotations/phantom39/
        xml_folder = os.path.join(ANNOTATION_DIR, video_name)

        if not os.path.exists(xml_folder):
            print(f"⚠️ 跳过: 找不到标注文件夹 {xml_folder}")
            continue

        cap = cv2.VideoCapture(v_path)
        system.reset_state()

        # 获取总帧数用于进度条
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        desc_str = f"Processing {video_name}"
        pbar = tqdm(total=total_frames, desc=desc_str, leave=False)

        frame_idx = 1  # XML 通常从 1 开始计数，或者从 0，需要检查你的数据

        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. 运行系统
            pred = system.process_frame(frame)

            # 2. 读取对应 XML
            # ARD-MAV 格式: phantom39_0001.xml
            # 注意: 这里假设从 0001 开始。如果报错，请检查是否是 0000 开始
            xml_filename = f"{video_name}_{frame_idx:04d}.xml"
            xml_path = os.path.join(xml_folder, xml_filename)

            gts = []
            if os.path.exists(xml_path):
                gts = parse_xml_gt(xml_path)
            else:
                # 某些数据集可能中间缺帧没有 XML（通常意味着没有目标）
                # 或者文件名格式不对
                pass

                # 3. 记录结果
            evaluator.add_batch(pred, gts)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

    print("\n" + "=" * 40)
    print("�� 最终评估报告 (All Videos)")
    print("=" * 40)

    # 计算 mAP@0.5
    p50, r50, ap50 = evaluator.compute_metrics(iou_thresh=0.5)
    print(f"Precision : {p50:.4f}")
    print(f"Recall    : {r50:.4f}")
    print(f"mAP@0.5   : {ap50:.4f}")

    # 计算 mAP@0.5:0.95
    ap_sum = 0
    thresholds = np.arange(0.5, 1.0, 0.05)
    print("\n正在计算 mAP@0.5:0.95 ...")
    for thresh in thresholds:
        _, _, ap = evaluator.compute_metrics(iou_thresh=thresh)
        ap_sum += ap
    map_50_95 = ap_sum / len(thresholds)

    print(f"mAP@0.5:0.95 : {map_50_95:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()