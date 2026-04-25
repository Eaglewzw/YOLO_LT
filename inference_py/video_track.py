import cv2
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.tracker.lighttrack import Lighttrack
from lib.utils.utils import get_subwindow_tracking

# ==================== Configuration ====================
VIDEO_PATH = "/home/verse/Videos/phantom13.mp4"
OUTPUT_PATH = "./output_tracked.mp4"

INIT_MODEL_PATH = "./model/ligthtrack_init.pt"    # 你的 init 模型
UPDATE_MODEL_PATH = "./model/ligthtrack_update.pt"  # 你的 update 模型

EXEMPLAR_SIZE = 127
INSTANCE_SIZE = 288
TOTAL_STRIDE = 16
SCORE_SIZE = 18

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== Load Models ====================
print(f"Loading models on {DEVICE}...")
init_model = torch.jit.load(INIT_MODEL_PATH, map_location=DEVICE)
update_model = torch.jit.load(UPDATE_MODEL_PATH, map_location=DEVICE)
init_model.eval()
update_model.eval()

# ==================== Tracker Config ====================
class DummyInfo:
    def __init__(self):
        self.stride = TOTAL_STRIDE
        self.dataset = 'CUSTOM'

info = DummyInfo()
tracker = Lighttrack(info, even=0)


# penalty_k: 0.007
# lr: 0.616
# window_influence: 0.225
# small_sz: 256
# big_sz: 288
# ratio: 1

p = type('P', (), {})()
p.exemplar_size = EXEMPLAR_SIZE
p.instance_size = INSTANCE_SIZE
p.total_stride = TOTAL_STRIDE
p.score_size = SCORE_SIZE
p.context_amount = 0.5
p.penalty_k = 0.007
p.window_influence = 0.225
p.lr = 0.616
p.windowing = 'cosine'

tracker.grids(p)
print(f"Grid shape: {tracker.grid_to_search_x.shape}")

# ==================== Init Template ====================
window = None  # 全局

def init_template(frame, target_pos, target_sz):
    global window

    wc_z = target_sz[0] + p.context_amount * (target_sz[0] + target_sz[1])
    hc_z = target_sz[1] + p.context_amount * (target_sz[0] + target_sz[1])
    s_z = round(np.sqrt(wc_z * hc_z))

    avg_chans = np.mean(frame, axis=(0, 1))
    z_crop, _ = get_subwindow_tracking(frame, target_pos, p.exemplar_size, s_z, avg_chans)

    z_crop = z_crop.float()  # 关键！
    # BGR → RGB
    z_crop_rgb = z_crop[[2,1,0], :, :]  # 在 CHW 上直接交换通道
    z_tensor = z_crop_rgb.to(DEVICE).unsqueeze(0)



    # 计算具体的数值
    mean_vals = torch.tensor([
        0.485 * 255.0,  # R通道均值
        0.456 * 255.0,  # G通道均值
        0.406 * 255.0   # B通道均值
    ], device=z_tensor.device)

    norm_vals = torch.tensor([
        1.0 / (0.229 * 255.0),  # R通道归一化系数
        1.0 / (0.224 * 255.0),  # G通道归一化系数
        1.0 / (0.225 * 255.0)   # B通道归一化系数
    ], device=z_tensor.device)

    mean_vals = mean_vals.view(3, 1, 1)
    norm_vals = norm_vals.view(3, 1, 1)

    # 应用ncnn风格的归一化: (x - mean) * norm
    z_tensor = (z_tensor - mean_vals) * norm_vals

    # 打印归一化后的范围用于调试
    print(f"归一化后范围: [{z_tensor.min().item():.3f}, {z_tensor.max().item():.3f}]")

    with torch.no_grad():
        zf = init_model(z_tensor)  # (1,96,8,8)

    print(f"zf shape: {zf.shape}")        # 应为 [1, 96, 8, 8] 或类似
    print(f"zf range: [{zf.min():.1f}, {zf.max():.1f}]")  # 应为 [-50, 150]

    # --- 生成 Hanning Window（与 C++ 一致）---
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(p.score_size) / (p.score_size - 1))
    window = np.outer(hanning, hanning)  # (18,18)

    print(f"[INIT] zf: {zf.shape}, window: {window.shape}")
    print(f"[INIT] z_tensor range: [{z_tensor.min():.2f}, {z_tensor.max():.2f}]")
    return zf

# ==================== Safe Track Frame ====================
def track_frame(frame, target_pos, target_sz, zf):
    # --- 1. Compute search region ---
    wc_z = target_sz[0] + p.context_amount * (target_sz[0] + target_sz[1])
    hc_z = target_sz[1] + p.context_amount * (target_sz[0] + target_sz[1])
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    avg_chans = np.mean(frame, axis=(0, 1))
    x_crop, _ = get_subwindow_tracking(frame, target_pos, p.instance_size, round(s_x), avg_chans)

    # BGR → RGB
    z_crop_rgb = x_crop[[2,1,0], :, :]  # 在 CHW 上直接交换通道
    z_tensor = z_crop_rgb.to(DEVICE).unsqueeze(0)


    # 计算具体的数值
    mean_vals = torch.tensor([
        0.485 * 255.0,  # R通道均值
        0.456 * 255.0,  # G通道均值
        0.406 * 255.0   # B通道均值
    ], device=z_tensor.device)

    norm_vals = torch.tensor([
        1.0 / (0.229 * 255.0),  # R通道归一化系数
        1.0 / (0.224 * 255.0),  # G通道归一化系数
        1.0 / (0.225 * 255.0)   # B通道归一化系数
    ], device=z_tensor.device)

    mean_vals = mean_vals.view(3, 1, 1)
    norm_vals = norm_vals.view(3, 1, 1)

    # 应用ncnn风格的归一化: (x - mean) * norm
    z_tensor = (z_tensor - mean_vals) * norm_vals

    # 打印归一化后的范围用于调试
    print(f"归一化后范围: [{z_tensor.min().item():.3f}, {z_tensor.max().item():.3f}]")

    # --- 2. Forward ---
    with torch.no_grad():
        cls, bbox = update_model(zf, z_tensor)
    print(f"cls_raw range: [{cls.min():.3f}, {cls.max():.3f}]")  # 应 ≈ [-5, 5]
    print(f"bbox range: [{bbox.min():.3f}, {bbox.max():.3f}]")  # 应 ≈ [-5, 5]


    # --- 3. Process outputs (align with C++: full squeeze) ---
    cls_score = torch.sigmoid(cls).squeeze().cpu().numpy()  # (18,18)
    bbox_pred = bbox.squeeze().cpu().numpy()               # (4,18,18)


    # --- 4. Align with C++ update processing ---
    cols = p.score_size  # 18
    rows = p.score_size  # 18

    # Split bbox channels
    bbox_data1 = bbox_pred[0]  # left
    bbox_data2 = bbox_pred[1]  # top
    bbox_data3 = bbox_pred[2]  # right
    bbox_data4 = bbox_pred[3]  # bottom

    # Compute pred
    pred_x1 = tracker.grid_to_search_x - bbox_data1
    pred_y1 = tracker.grid_to_search_y - bbox_data2
    pred_x2 = tracker.grid_to_search_x + bbox_data3
    pred_y2 = tracker.grid_to_search_y + bbox_data4

    # Compute w, h
    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1

    eps = 1e-6
    pred_w = np.clip(pred_w, eps, None)
    pred_h = np.clip(pred_h, eps, None)

    # sz_wh = sz_whFun(target_sz) in C++
    def sz_wh_fun(w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    sz_wh_target = sz_wh_fun(target_sz[0], target_sz[1])

    s_c = np.maximum(sz_wh_fun(pred_w, pred_h) / sz_wh_target, 1.0 / (sz_wh_fun(pred_w, pred_h) / sz_wh_target))

    ratio = target_sz[0] / (target_sz[1] + eps)
    pred_ratio = pred_w / (pred_h + eps)
    r_c = np.maximum(ratio / pred_ratio, pred_ratio / ratio)

    # penalty
    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)

    # pscore
    pscore = (penalty * cls_score) * (1 - p.window_influence) + window * p.window_influence

    # Safe argmax
    if np.isnan(pscore).any() or np.isinf(pscore).any():
        print("[WARN] pscore contains nan/inf, fallback to center")
        r_max, c_max = rows // 2, cols // 2
    else:
        idx = np.argmax(pscore)
        r_max = idx // cols
        c_max = idx % cols

    # pred real
    pred_x1_real = pred_x1[r_max, c_max]
    pred_y1_real = pred_y1[r_max, c_max]
    pred_x2_real = pred_x2[r_max, c_max]
    pred_y2_real = pred_y2[r_max, c_max]

    pred_xs = (pred_x1_real + pred_x2_real) / 2
    pred_ys = (pred_y1_real + pred_y2_real) / 2
    pred_w = pred_x2_real - pred_x1_real
    pred_h = pred_y2_real - pred_y1_real

    diff_xs = pred_xs - p.instance_size // 2
    diff_ys = pred_ys - p.instance_size // 2
    diff_xs /= scale_z
    diff_ys /= scale_z
    # pred_w /= scale_z
    # pred_h /= scale_z

    target_sz_scaled = target_sz * scale_z
    lr_ = np.clip(penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr, 0, 1)

    res_xs = target_pos[0] + diff_xs
    res_ys = target_pos[1] + diff_ys
    res_w = pred_w * lr_ + (1 - lr_) * target_sz_scaled[0]
    res_h = pred_h * lr_ + (1 - lr_) * target_sz_scaled[1]

    target_pos = np.array([res_xs, res_ys])
    target_sz = target_sz_scaled * (1 - lr_) + lr_ * np.array([res_w, res_h])
    target_sz /= scale_z

    # Clamp
    im_h, im_w = frame.shape[:2]
    target_pos = np.clip(target_pos, [0, 0], [im_w, im_h])
    target_sz = np.clip(target_sz, 10, [im_w, im_h])

    overlay = visualize_response_cv(frame, cls_score, r_max, c_max, scale_z, target_pos, target_sz)
    cv2.namedWindow("LightTrack - Heatmap", cv2.WINDOW_NORMAL)  # 必须
    cv2.resizeWindow("LightTrack - Heatmap", 640, 640)  # 最大 1000x700
    cv2.imshow("LightTrack - Heatmap", overlay)

    return target_pos, target_sz

# ==================== Main ====================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        print("Unable to read video")
        return

    print("Select target and press ENTER...")
    cv2.namedWindow("Select", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select", 640, 640)  # 最大 1000x700
    bbox = cv2.selectROI("Select", frame, False)
    # bbox =[249, 157, 102, 94]
    x, y, w, h = bbox
    target_pos = np.array([x + w/2, y + h/2])
    target_sz = np.array([w, h])

    cropped = frame[y:y+h, x:x+w]  # 注意：先 y 后 x！
    cv2.namedWindow("Select", cv2.WINDOW_NORMAL)  # 必须
    cv2.imshow("Select", cropped)

    cv2.namedWindow("LightTrack Demo", cv2.WINDOW_NORMAL)  # 必须
    cv2.resizeWindow("LightTrack Demo", 640, 640)  # 最大 1000x700

    zf = init_template(frame, target_pos, target_sz)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx == 1:
            continue

        target_pos, target_sz = track_frame(frame, target_pos, target_sz, zf)

        pos = target_pos.astype(int)
        sz = target_sz.astype(int)
        x1 = pos[0] - sz[0]//2
        y1 = pos[1] - sz[1]//2
        x2 = pos[0] + sz[0]//2
        y2 = pos[1] + sz[1]//2

        # cropped = frame[y1:y2, x1:x2]  # 注意：先 y 后 x！
        # cv2.imshow("Track", cropped)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("LightTrack Demo", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Tracking complete! Saved to: {OUTPUT_PATH}")
def visualize_response_cv(frame, cls_score, r_max, c_max, scale_z, target_pos, target_sz):
    im_h, im_w = frame.shape[:2]

    # 1. 归一化 + 上采样到 288x288
    cls_norm = (cls_score - cls_score.min()) / (cls_score.max() - cls_score.min() + 1e-8)
    cls_vis = (cls_norm * 255).astype(np.uint8)
    heatmap_full = cv2.resize(cls_vis, (p.instance_size, p.instance_size))  # 288x288
    heatmap_full = cv2.applyColorMap(heatmap_full, cv2.COLORMAP_JET)  # BGR

    # 2. 计算搜索区域在原图中的位置
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z_inv = s_z / p.exemplar_size

    pad = (p.instance_size - p.exemplar_size) / 2.0 / scale_z_inv
    cx, cy = target_pos

    x1 = int(cx - s_z / 2 - pad)
    y1 = int(cy - s_z / 2 - pad)
    x2 = int(cx + s_z / 2 + pad)
    y2 = int(cy + s_z / 2 + pad)

    # 3. 边界裁剪
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(im_w, x2)
    y2 = min(im_h, y2)

    h_crop, w_crop = y2 - y1, x2 - x1
    if h_crop <= 0 or w_crop <= 0:
        return frame.copy()

    # 4. 关键修复：resize heatmap 到实际裁剪尺寸
    heatmap_crop = cv2.resize(heatmap_full, (w_crop, h_crop))

    # 5. 叠加（现在尺寸一致！）
    overlay = frame.copy()
    alpha = 0.5
    overlay[y1:y2, x1:x2] = cv2.addWeighted(
        heatmap_crop, alpha,
        overlay[y1:y2, x1:x2], 1 - alpha, 0
    )

    # 6. 峰值映射（比例缩放）
    scale_x = w_crop / p.instance_size
    scale_y = h_crop / p.instance_size
    peak_x_local = c_max * scale_x
    peak_y_local = r_max * scale_y
    peak_x_global = int(x1 + peak_x_local)
    peak_y_global = int(y1 + peak_y_local)

    cv2.drawMarker(overlay, (peak_x_global, peak_y_global),
                   color=(0, 255, 255), markerType=cv2.MARKER_CROSS,
                   markerSize=20, thickness=3)

    # 7. 画框
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
    tx1 = int(cx - target_sz[0]/2)
    ty1 = int(cy - target_sz[1]/2)
    tx2 = int(cx + target_sz[0]/2)
    ty2 = int(cy + target_sz[1]/2)
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

    # 8. 文字
    cv2.putText(overlay, f"Peak: ({r_max},{c_max})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Score: {cls_score.max():.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return overlay

if __name__ == "__main__":
    main()