import cv2
import os
import numpy as np
import json
import pandas as pd
import xml.etree.ElementTree as ET
import time
import warnings
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
warnings.filterwarnings("ignore")

# ================= 1. 路径与模型配置 =================
IMAGE_ROOT = "/media/verser/robot/Dataset/ARD-MAV/Images"
ANNO_ROOT  = "/media/verser/robot/Dataset/ARD-MAV/Annotations"
YOLO_ENGINE_PATH = "./model/TensorRT_10/yolov5s_GLAD.pt"
INIT_MODEL  = "./model/ligthtrack_init.pt"
UPDATE_MODEL = "./model/ligthtrack_update.pt"
DEVICE = 'cuda'

VISUAL_FAIL_THRESHOLD = 30
MOD_FAIL_THRESHOLD    = 10

# ================= 2. 消融配置表 =================
ABLATION_CONFIGS = [
    {"name": "Full",      "VISUAL_DETECT": True,  "MOTION_DETECT": True,  "TRACKING": True},
    {"name": "w/o MOD",   "VISUAL_DETECT": True,  "MOTION_DETECT": False, "TRACKING": True},
    {"name": "w/o YOLO",  "VISUAL_DETECT": False, "MOTION_DETECT": True,  "TRACKING": True},
    {"name": "w/o LT",    "VISUAL_DETECT": True,  "MOTION_DETECT": True,  "TRACKING": False},
    {"name": "Only YOLO", "VISUAL_DETECT": True,  "MOTION_DETECT": False, "TRACKING": False},
    {"name": "Only MOD",  "VISUAL_DETECT": False, "MOTION_DETECT": True,  "TRACKING": False},
    {"name": "YOLO+LT",   "VISUAL_DETECT": True,  "MOTION_DETECT": False, "TRACKING": True},
]

# ================= 3. 模块导入 =================
try:
    from modules_trt10.YOLO_Engine_TRT10 import YOLO_Detector
    from LightTrack_FastTrack import LightTrackEngine
    from MOD2 import MOD2_global
    print("✅ 系统核心模块加载成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# ================= 4. 辅助函数 =================
def get_safe_roi(frame, bbox, expansion_factor=3.0):
    if bbox is None: return None, None
    img_h, img_w = frame.shape[:2]
    x, y, w, h = map(int, bbox[:4])
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * expansion_factor, h * expansion_factor
    x1, y1 = int(max(0, cx - nw / 2)), int(max(0, cy - nh / 2))
    x2, y2 = int(min(img_w, cx + nw / 2)), int(min(img_h, cy + nh / 2))
    roi = frame[y1:y2, x1:x2]
    return (roi, (x1, y1, x2, y2)) if roi.size > 0 else (None, None)


def calculate_iou(boxA, boxB):
    if boxA == [0, 0, 0, 0] or boxB == [0, 0, 0, 0]: return 0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA, areaB = boxA[2] * boxA[3], boxB[2] * boxB[3]
    return inter / float(areaA + areaB - inter + 1e-6)


# ================= 5. 评估器 =================
class ARD_Ablation_Evaluator:
    def __init__(self, seq_name):
        self.seq_name = seq_name
        self.img_dir  = os.path.join(IMAGE_ROOT, seq_name)
        self.anno_dir = os.path.join(ANNO_ROOT,  seq_name)

        if not os.path.exists(self.img_dir):
            self.num_frames = 0
            return

        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        self.gt_bboxes = self._load_voc_xmls()
        self.num_frames = min(len(self.img_files), len(self.gt_bboxes))

    def _load_voc_xmls(self):
        xml_files = sorted([f for f in os.listdir(self.anno_dir) if f.endswith('.xml')])
        gt_list = []
        for x_file in xml_files:
            try:
                tree = ET.parse(os.path.join(self.anno_dir, x_file))
                root = tree.getroot()
                found = False
                for obj in root.findall('object'):
                    if obj.find('name').text.lower() in ['drone', 'uav']:
                        b = obj.find('bndbox')
                        xmin, ymin = float(b.find('xmin').text), float(b.find('ymin').text)
                        xmax, ymax = float(b.find('xmax').text), float(b.find('ymax').text)
                        gt_list.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        found = True; break
                if not found: gt_list.append([0, 0, 0, 0])
            except:
                gt_list.append([0, 0, 0, 0])
        return gt_list

    def run_inference(self, yolo, tracker, config):
        """
        config 字段:
          VISUAL_DETECT : bool  是否使用 YOLO
          MOTION_DETECT : bool  是否使用 MOD2
          TRACKING      : bool  是否使用 LightTrack
        """
        use_yolo    = config["VISUAL_DETECT"]
        use_mod     = config["MOTION_DETECT"]
        use_tracker = config["TRACKING"]

        tracking_state = False
        v_fail_cnt, m_fail_cnt = 0, 0
        prev_frame = None

        raw_preds          = []
        inference_durations = []
        correct_frames     = 0

        for i in tqdm(range(self.num_frames), desc=f"  {self.seq_name}", leave=False):
            curr_frame = cv2.imread(os.path.join(self.img_dir, self.img_files[i]))
            if curr_frame is None: continue

            t_start = time.perf_counter()
            bbox, score = None, 0.0

            # ---------- 搜索阶段 ----------
            if not tracking_state:
                # 策略 A: YOLO
                if use_yolo and v_fail_cnt < VISUAL_FAIL_THRESHOLD:
                    bbox = yolo.detect(curr_frame)
                    if bbox is not None:
                        v_fail_cnt, m_fail_cnt = 0, 0
                    else:
                        v_fail_cnt += 1

                # 策略 B: MOD2 (YOLO 未开 或 YOLO 连续失败)
                if bbox is None and use_mod and prev_frame is not None:
                    res = MOD2_global(prev_frame, curr_frame)
                    if res:
                        bbox = list(res[0]) if isinstance(res[0], (list, np.ndarray)) else list(res)
                        m_fail_cnt = 0
                    else:
                        m_fail_cnt += 1
                        if m_fail_cnt >= MOD_FAIL_THRESHOLD:
                            v_fail_cnt, m_fail_cnt = 0, 0

                # 初始化追踪器
                if bbox is not None and use_tracker:
                    tracker.init(curr_frame, bbox[:4])
                    tracking_state, score = True, 1.0

            # ---------- 追踪阶段 ----------
            else:
                bbox, score = tracker.track(curr_frame)

                # 周期性 ROI 校验 (仅当 YOLO 开启时)
                if use_yolo and i % 60 == 0:
                    roi, _ = get_safe_roi(curr_frame, bbox)
                    if roi is not None:
                        rb = yolo.detect(roi)
                        if rb is None or rb[4] < 0.45:
                            tracking_state, score = False, 0.0

                if score < 0.98:
                    tracking_state, v_fail_cnt = False, 0

            t_end = time.perf_counter()
            inference_durations.append(t_end - t_start)

            pred = [float(x) for x in bbox[:4]] if (bbox is not None and (tracking_state or not use_tracker)) else [0, 0, 0, 0]
            raw_preds.append(pred)

            if calculate_iou(self.gt_bboxes[i], pred) >= 0.5:
                correct_frames += 1

            prev_frame = curr_frame.copy()

        return self._compute_metrics(raw_preds, inference_durations, correct_frames)

    def _compute_metrics(self, raw_preds, durations, correct_frames):
        fps = 1.0 / np.mean(durations) if durations else 0
        cdr = (correct_frames / self.num_frames) * 100 if self.num_frames > 0 else 0

        ious = [calculate_iou(gt, pr) for gt, pr in zip(self.gt_bboxes, raw_preds)]
        tp = np.sum(np.array(ious) >= 0.5)
        total_pred = np.sum(np.array([sum(p) for p in raw_preds]) > 0)
        total_gt   = len([g for g in self.gt_bboxes if sum(g) > 0])

        p  = tp / total_pred if total_pred > 0 else 0
        r  = tp / total_gt   if total_gt   > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        auc = np.mean([np.sum(np.array(ious) > t) / len(ious) for t in np.linspace(0, 1, 50)])

        # ---------- AP@0.5 (pycocotools) ----------
        ap50 = self._compute_ap50(raw_preds)

        return {
            "Precision":    round(p,    4),
            "Recall":       round(r,    4),
            "F1-score":     round(f1,   4),
            "AP@0.5":       round(ap50, 4),
            "Success(AUC)": round(auc,  4),
            "FPS":          round(fps,  2),
            "CDR (%)":      round(cdr,  2),
        }

    def _compute_ap50(self, raw_preds):
        # 构建 COCO 格式 GT
        coco_gt_dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "drone"}]
        }
        coco_dt_list = []
        ann_id = 0
        for i, (gt, pred) in enumerate(zip(self.gt_bboxes, raw_preds)):
            coco_gt_dict["images"].append({"id": i})
            if sum(gt) > 0:
                coco_gt_dict["annotations"].append({
                    "id": ann_id, "image_id": i, "category_id": 1,
                    "bbox": gt, "area": gt[2] * gt[3], "iscrowd": 0
                })
                ann_id += 1
            if sum(pred) > 0:
                iou_score = calculate_iou(gt, pred) if sum(gt) > 0 else 0.0
                coco_dt_list.append({
                    "image_id": i, "category_id": 1,
                    "bbox": pred, "score": float(iou_score)
                })

        if not coco_gt_dict["annotations"] or not coco_dt_list:
            return 0.0

        import tempfile, json as _json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            _json.dump(coco_gt_dict, f)
            gt_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            _json.dump(coco_dt_list, f)
            dt_path = f.name

        coco_gt   = COCO(gt_path)
        coco_dt   = coco_gt.loadRes(dt_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.iouThrs = np.array([0.5])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        os.unlink(gt_path)
        os.unlink(dt_path)
        return float(coco_eval.stats[0])


# ================= 6. 主程序 =================
if __name__ == "__main__":
    yolo    = YOLO_Detector(YOLO_ENGINE_PATH)
    tracker = LightTrackEngine(INIT_MODEL, UPDATE_MODEL, device=DEVICE)

    sequences = sorted([
        d for d in os.listdir(IMAGE_ROOT)
        if os.path.isdir(os.path.join(IMAGE_ROOT, d))
    ])

    all_rows = []

    for cfg in ABLATION_CONFIGS:
        cfg_name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"▶ 消融配置: [{cfg_name}]  "
              f"YOLO={cfg['VISUAL_DETECT']}  "
              f"MOD={cfg['MOTION_DETECT']}  "
              f"LT={cfg['TRACKING']}")
        print('='*60)

        cfg_metrics = []
        for seq in sequences:
            ev = ARD_Ablation_Evaluator(seq)
            if ev.num_frames == 0:
                print(f"  ⚠️ 跳过 {seq}: 路径无效")
                continue

            metrics = ev.run_inference(yolo, tracker, cfg)
            metrics["Config"]   = cfg_name
            metrics["Sequence"] = seq
            cfg_metrics.append(metrics)
            all_rows.append(metrics)

            print(f"  ✔ {seq:20s} | F1={metrics['F1-score']:.4f} "
                  f"| AP@0.5={metrics['AP@0.5']:.4f} "
                  f"| AUC={metrics['Success(AUC)']:.4f} "
                  f"| CDR={metrics['CDR (%)']:.1f}% "
                  f"| FPS={metrics['FPS']:.1f}")

        # 每个配置打印均值
        if cfg_metrics:
            df_cfg = pd.DataFrame(cfg_metrics)
            mean = df_cfg.mean(numeric_only=True)
            print(f"  {'MEAN':20s} | F1={mean['F1-score']:.4f} "
                  f"| AP@0.5={mean['AP@0.5']:.4f} "
                  f"| AUC={mean['Success(AUC)']:.4f} "
                  f"| CDR={mean['CDR (%)']:.1f}% "
                  f"| FPS={mean['FPS']:.1f}")

    # ---- 汇总表 ----
    df_all = pd.DataFrame(all_rows)
    cols = ["Config", "Sequence", "Precision", "Recall", "F1-score", "AP@0.5", "Success(AUC)", "FPS", "CDR (%)"]
    df_all = df_all[cols]

    # 每个 Config 的均值行
    mean_rows = []
    for cfg_name, grp in df_all.groupby("Config", sort=False):
        m = grp.mean(numeric_only=True).to_dict()
        m["Config"] = cfg_name
        m["Sequence"] = "AVERAGE"
        mean_rows.append(m)

    df_mean = pd.DataFrame(mean_rows)[cols]

    print("\n" + "="*70)
    print("📊 ARD-MAV 消融实验汇总 (各配置均值)")
    print("="*70)
    print(df_mean.to_string(index=False))

    out_path = "ablation_ARD_results.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n✅ 完整结果已保存至: {out_path}")
