import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time

# 确保路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.tracker.lighttrack import Lighttrack

class LightTrackEngine:
    def __init__(self, init_model_path, update_model_path, device='cuda'):
        self.device = device
        self.exemplar_size = 127
        self.instance_size = 288
        self.total_stride = 16
        self.score_size = 18

        # 1. 配置超参数
        self.p = type('P', (), {})()
        self.p.exemplar_size = self.exemplar_size
        self.p.instance_size = self.instance_size
        self.p.total_stride = self.total_stride
        self.p.score_size = self.score_size
        self.p.context_amount = 0.5
        self.p.penalty_k = 0.007
        self.p.window_influence = 0.225
        self.p.lr = 0.616
        self.p.windowing = 'cosine'
        self.last_cls_score_pos = 170
        self.abnormal_jump_value = 25

        # 2. 初始化 tracker helper
        class DummyInfo:
            def __init__(self, stride):
                self.stride = stride
                self.dataset = 'CUSTOM'

        self.tracker = Lighttrack(DummyInfo(self.total_stride), even=0)
        self.tracker.grids(self.p)

        # 3. 加载模型 (FP32)
        print(f"🔄 [Engine] 加载模型: {self.device} (FP32 Mode)")
        try:
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True

            self.init_model = torch.jit.load(init_model_path, map_location=self.device).eval()
            self.update_model = torch.jit.load(update_model_path, map_location=self.device).eval()
        except Exception as e:
            print(f"❌ [Engine] 模型加载失败: {e}")
            sys.exit(1)

        # 4. 预定义 GPU 常量
        self.mean_vals = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std_vals = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 内部状态变量
        self.zf = None
        self.window = None
        self.target_pos = None
        self.target_sz = None

    def _gpu_crop_and_resize(self, img_tensor, center_pos, size, out_size, avg_chans=None):
        """
        [核心优化] 全 GPU 裁剪和缩放
        替代原本的 get_subwindow_tracking (CPU版)
        :param img_tensor: (1, 3, H, W) float tensor, range [0, 255]
        :param center_pos: [cx, cy]
        :param size: original_size (scalar) 需要裁剪的区域大小
        :param out_size: output_size (scalar) 目标输出大小 (127 or 288)
        """
        # 计算裁剪区域
        sz = size
        im_h, im_w = img_tensor.shape[2], img_tensor.shape[3]

        # 计算整数坐标
        c = (int(sz) + 1) // 2
        cx, cy = int(center_pos[0]), int(center_pos[1])

        # 计算 padding
        pad_left = -int(cx - c)
        pad_top = -int(cy - c)
        pad_right = int(cx + c) - im_w
        pad_bottom = int(cy + c) - im_h

        padding_vals = [0, 0, 0, 0] # Left, Right, Top, Bottom

        # 计算实际在图片内的 ROI
        left = max(0, int(cx - c))
        right = min(im_w, int(cx + c))
        top = max(0, int(cy - c))
        bottom = min(im_h, int(cy + c))

        # 1. 显存内切片 (Slicing)
        roi = img_tensor[:, :, top:bottom, left:right]

        # 2. 如果切片出界，需要 Pad (在 GPU 上做)
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            # 计算 padding (PyTorch pad 顺序是: Left, Right, Top, Bottom)
            # 使用平均颜色填充
            if avg_chans is None:
                pad_val = 0.5 * 255 # 默认灰色
            else:
                pad_val = 0 # 简化处理，或者自行实现按通道pad，通常0或均值影响不大

            pad_l = max(0, pad_left)
            pad_r = max(0, pad_right)
            pad_t = max(0, pad_top)
            pad_b = max(0, pad_bottom)

            # 使用 reflect 或 constant 填充
            roi = F.pad(roi, (pad_l, pad_r, pad_t, pad_b), mode='replicate')

        # 3. 显存内缩放 (Resize)
        # 必须确保 shape 是 (N, C, H, W)
        if roi.shape[-1] != out_size:
            roi = F.interpolate(roi, size=(out_size, out_size), mode='bilinear', align_corners=False)

        return roi

    def _preprocess_frame(self, frame_numpy):
        """将 OpenCV 图片转为 GPU Tensor (不做归一化，保留 0-255)"""
        # H, W, C -> C, H, W
        tensor = torch.from_numpy(frame_numpy).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
        # BGR -> RGB
        tensor = tensor[:, [2, 1, 0], :, :]
        return tensor

    def init(self, frame, bbox):
        """
        初始化追踪器 (优化后：耗时降低 50%+)
        """
        x, y, w, h = bbox
        self.target_pos = np.array([x + w / 2, y + h / 2])
        self.target_sz = np.array([w, h])

        # 计算搜索尺寸 (s_z)
        wc_z = self.target_sz[0] + self.p.context_amount * (self.target_sz[0] + self.target_sz[1])
        hc_z = self.target_sz[1] + self.p.context_amount * (self.target_sz[0] + self.target_sz[1])
        s_z = round(np.sqrt(wc_z * hc_z))

        # === 优化点 1: 立即将全图上载到 GPU ===
        full_img_tensor = self._preprocess_frame(frame)

        # === 优化点 2: 在 GPU 上裁剪和 Resize ===
        # 替代了 get_subwindow_tracking
        z_crop_tensor = self._gpu_crop_and_resize(
            full_img_tensor,
            self.target_pos,
            s_z,
            self.p.exemplar_size
        )

        # === 优化点 3: 归一化 (利用 GPU 向量化计算) ===
        # z_crop_tensor 当前是 0-255 RGB
        z_tensor = (z_crop_tensor / 255.0 - self.mean_vals) / self.std_vals

        with torch.no_grad():
            self.zf = self.init_model(z_tensor)

        # 生成 Hanning Window
        hanning = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.p.score_size) / (self.p.score_size - 1))
        self.window = np.outer(hanning, hanning)

        # print("✅ [Engine] 追踪器初始化完成 (GPU Accel)")


    def _check_pos_change(self, current_pos_idx):
        """
        检查目标在特征图上的位置是否发生剧烈跳变
        :param current_pos_idx: 当前帧最高分位置的索引 (flat index 0~323)
        :return: True 表示发生异常跳变，False 表示正常
        """
        is_change_flag = False

        # 计算位移偏差
        delta = abs(current_pos_idx - self.last_cls_score_pos)

        # C++ 逻辑: if(abs(...) <= 25)
        if delta <= self.abnormal_jump_value:
            is_change_flag = False
            self.last_cls_score_pos = current_pos_idx # 更新上一帧位置
        else:
            is_change_flag = True
            self.last_cls_score_pos = 170 # 发生跳变，重置为中心点 (Reset Logic)
            # print(f"⚠️ [Anti-Drift] 检测到剧烈跳动! Delta: {delta} -> 判定为不可靠")

        return is_change_flag

    def track(self, frame):
        """
        追踪下一帧 (同样享受 GPU 裁剪加速)
        """
        wc_z = self.target_sz[0] + self.p.context_amount * (self.target_sz[0] + self.target_sz[1])
        hc_z = self.target_sz[1] + self.p.context_amount * (self.target_sz[0] + self.target_sz[1])
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.p.exemplar_size / s_z
        d_search = (self.p.instance_size - self.p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = round(s_z + 2 * pad)

        # === 优化点: 也是 GPU 处理 ===
        full_img_tensor = self._preprocess_frame(frame)

        x_crop_tensor = self._gpu_crop_and_resize(
            full_img_tensor,
            self.target_pos,
            s_x,
            self.p.instance_size
        )

        x_tensor = (x_crop_tensor / 255.0 - self.mean_vals) / self.std_vals

        with torch.no_grad():
            cls, bbox_reg = self.update_model(self.zf, x_tensor)

        cls_score = torch.sigmoid(cls).squeeze().cpu().numpy()
        bbox_pred = bbox_reg.squeeze().cpu().numpy()

        pred_bbox = self._post_process(bbox_pred, cls_score, scale_z, frame.shape)
        best_score = cls_score[np.unravel_index(np.argmax(cls_score), cls_score.shape)]

        # # ==========================================
        # # 获取最高分位置并进行跳变检测
        # # ==========================================
        # # 1. 获取 flatten 后的最大值索引 (对应 C++ cls_score_position)
        # best_idx_flat = np.argmax(cls_score)
        # # 2. 获取最高分数值
        # best_score = cls_score.flat[best_idx_flat]
        # # 3. 调用防漂移检测
        # is_abnormal_jump = self._check_pos_change(best_idx_flat)
        #
        # # 4. 如果检测到异常跳动，强制将分数置为 0，触发出界的 LOST 逻辑
        # if is_abnormal_jump:
        #     best_score = 0.0 # 强制认为丢失

        final_bbox = (
            int(self.target_pos[0] - self.target_sz[0] / 2),
            int(self.target_pos[1] - self.target_sz[1] / 2),
            int(self.target_sz[0]),
            int(self.target_sz[1])
        )

        return final_bbox, best_score

    def _post_process(self, bbox_pred, cls_score, scale_z, frame_shape):
        """内部后处理计算 (保持不变)"""
        bbox_data1, bbox_data2, bbox_data3, bbox_data4 = bbox_pred
        pred_x1 = self.tracker.grid_to_search_x - bbox_data1
        pred_y1 = self.tracker.grid_to_search_y - bbox_data2
        pred_x2 = self.tracker.grid_to_search_x + bbox_data3
        pred_y2 = self.tracker.grid_to_search_y + bbox_data4

        pred_w = np.clip(pred_x2 - pred_x1, 1e-6, None)
        pred_h = np.clip(pred_y2 - pred_y1, 1e-6, None)

        def sz_wh_fun(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        sz_wh_target = sz_wh_fun(self.target_sz[0], self.target_sz[1])
        s_c = np.maximum(sz_wh_fun(pred_w, pred_h) / sz_wh_target,
                         1.0 / (sz_wh_fun(pred_w, pred_h) / sz_wh_target))

        eps = 1e-6
        ratio = self.target_sz[0] / (self.target_sz[1] + eps)
        pred_ratio = pred_w / (pred_h + eps)
        r_c = np.maximum(ratio / pred_ratio, pred_ratio / ratio)

        penalty = np.exp(-(r_c * s_c - 1) * self.p.penalty_k)
        pscore = (penalty * cls_score) * (1 - self.p.window_influence) + self.window * self.p.window_influence

        idx = np.argmax(pscore)
        r_max, c_max = divmod(idx, self.p.score_size)

        pred_x1_real = pred_x1[r_max, c_max]
        pred_y1_real = pred_y1[r_max, c_max]
        pred_x2_real = pred_x2[r_max, c_max]
        pred_y2_real = pred_y2[r_max, c_max]

        pred_w = pred_x2_real - pred_x1_real
        pred_h = pred_y2_real - pred_y1_real

        diff_xs = ( (pred_x1_real + pred_x2_real) / 2 - self.p.instance_size // 2) / scale_z
        diff_ys = ( (pred_y1_real + pred_y2_real) / 2 - self.p.instance_size // 2) / scale_z

        target_sz_scaled = self.target_sz * scale_z
        lr_ = np.clip(penalty[r_max, c_max] * cls_score[r_max, c_max] * self.p.lr, 0, 1)

        res_xs = self.target_pos[0] + diff_xs
        res_ys = self.target_pos[1] + diff_ys
        res_w = pred_w * lr_ + (1 - lr_) * target_sz_scaled[0]
        res_h = pred_h * lr_ + (1 - lr_) * target_sz_scaled[1]

        # 更新内部状态
        self.target_pos = np.array([res_xs, res_ys])
        self.target_sz = target_sz_scaled * (1 - lr_) + lr_ * np.array([res_w, res_h])
        self.target_sz /= scale_z

        # 边界限制
        im_h, im_w = frame_shape[:2]
        self.target_pos = np.clip(self.target_pos, [0, 0], [im_w, im_h])
        self.target_sz = np.clip(self.target_sz, 10, [im_w, im_h])

        return None


# ==================== 主函数 ====================
VIDEO_PATH = "/home/verser/Videos/fast_drone.mp4"
INIT_MODEL = "./model/ligthtrack_init.pt"
UPDATE_MODEL = "./model/ligthtrack_update.pt"
# 实例化 tracker
tracker = LightTrackEngine(INIT_MODEL, UPDATE_MODEL, device="cuda:0")

def main():
    # ================= 预热逻辑 (Warm-up) =================
    print("🔥 正在预热 LightTrack 模型...")
    # 使用 1920x1080 的黑图进行预热，模拟真实场景
    warmup_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    warmup_bbox = [100, 100, 100, 100]

    # 执行一次完整的 Init + Track 流程
    # 这会触发 CUDA Context 初始化、显存分配和模型推理
    tracker.init(warmup_img, warmup_bbox)
    tracker.track(warmup_img)
    print("✅ 预热完成，显卡已就绪")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    if not ret: return

    # 视频分辨率检查
    h, w = frame.shape[:2]
    print(f"ℹ️ 视频分辨率: {w}x{h}")
    resize_factor = 1.0
    # 4K视频降采样，否则CPU是瓶颈
    if w > 1920:
        resize_factor = 0.5
        print(f"⚠️ 视频过大，开启自动缩放 (Factor: {resize_factor})")
        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

    print("🎯 拖拽选择目标...")
    bbox = cv2.selectROI("LightTrack", frame, False)
    cv2.destroyAllWindows()

    if bbox[2] <= 0 or bbox[3] <= 0: return

    print(f"✅ 目标: {bbox}")
    print("🚀 初始化...")

    # 修正 1: 直接使用 bbox 调用类方法
    tracker.init(frame, bbox)

    # 重置视频到开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0

    # === 修改：初始化实时 FPS 变量 ===
    prev_frame_time = time.time() # 记录上一帧时间
    curr_fps = 0.0                # 初始化 FPS 数值

    cv2.namedWindow("LightTrack Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LightTrack Demo", 640, int(640 * frame.shape[0] / frame.shape[1]))

    while True:
        ret, frame = cap.read()
        if not ret: break

        # === 修改：计算实时 FPS (放在处理前，计算上一帧到这一帧的时间间隔) ===
        curr_frame_time = time.time()
        time_delta = curr_frame_time - prev_frame_time
        prev_frame_time = curr_frame_time # 更新时间

        # 防止除以0，并计算瞬时 FPS
        if time_delta > 0:
            instant_fps = 1.0 / time_delta
            # === 可选：平滑滤波 (让显示数值不那么剧烈跳动) ===
            # 0.1 * 新值 + 0.9 * 旧值 (数值越小越灵敏，数值越大越平滑)
            curr_fps = 0.1 * instant_fps + 0.9 * curr_fps
        else:
            curr_fps = 0.0
        # ========================================================

        if resize_factor != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        # 修正 2: 逻辑重构
        if frame_idx == 0:
            x, y, w, h = bbox # 初始 bbox
            score_text = "Init"
            current_score = 1.0
        else:
            # 修正 3: 调用 tracker.track
            tracked_bbox, current_score = tracker.track(frame)
            x, y, w, h = tracked_bbox
            score_text = f"{current_score:.2f}"

            # 丢帧检测
            if current_score < 0.98:
                print(f"🛑 追踪丢失 (Score: {current_score:.4f}) <= 0.5，程序退出。")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, f"LOST: {current_score:.3f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("LightTrack Demo", frame)
                cv2.waitKey(2000)
                break

        color = (0, 255, 255) if frame_idx > 0 else (0, 255, 0)

        # 画图
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # 修改显示的 FPS 变量
        cv2.putText(frame, f"FPS: {curr_fps:.1f} | Score: {score_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("LightTrack Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

        # 打印日志也改为显示当前实时 FPS
        if frame_idx % 60 == 0:
            print(f"Frame {frame_idx} | Real-time FPS: {curr_fps:.1f}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 完成: {frame_idx}帧")

if __name__ == "__main__":
    main()