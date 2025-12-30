import cv2
import time
import numpy as np
import sys
import os
from datetime import datetime  # 新增：用于生成时间戳文件名

# ==================== 导入自定义模块 ====================
# 请确保这些文件在同一目录下，或者在 PYTHONPATH 中
try:
    from MOD2 import MOD2_global
    # from LightTrack_Engine import LightTrackEngine
    from LightTrack_FastTrack import LightTrackEngine
    # 引用你修改过包含 fix 的 YOLO 类
    from YOLO_Engine_TensorRT10 import YOLO_Detector
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# ==================== 核心配置 ====================
# 1. 路径配置
VIDEO_PATH = "/home/verser/Videos/phantom13.mp4"
INIT_MODEL = "./model/TensorRT_10/ligthtrack_init.pt"
UPDATE_MODEL = "./model/TensorRT_10/ligthtrack_update.pt"
YOLO_ENGINE_PATH = "./model/TensorRT_10/yolov5s_GLAD.pt"

# 2. 设备配置
DEVICE = 'cuda'

# 3. 策略阈值
VISUAL_FAIL_THRESHOLD = 120

# 4. 功能开关
ENABLE_CONFIG = {
    "VISUAL_DETECT": True,  # 是否开启 YOLO 视觉检测
    "MOTION_DETECT": True,  # 是否开启 MOD2 运动检测
    "TRACKING": True,  # 是否开启 LightTrack 局部跟踪
    "SAVE_VIDEO": True  # 【新增】是否保存推理结果视频
}


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return

    # ==================== 【新增】视频保存初始化 ====================
    video_writer = None
    if ENABLE_CONFIG["SAVE_VIDEO"]:
        # 创建输出目录
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"result_{timestamp}.mp4")

        # 获取原视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化写入器 (使用 mp4v 编码)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f" 录像将保存至: {save_path}")
    # ==========================================================

    # ==================== 1. 模型初始化 (循环外) ====================

    # A. 初始化 LightTrack
    tracker = None
    if ENABLE_CONFIG["TRACKING"]:
        print(" 初始化 LightTrack 追踪器...")
        if os.path.exists(INIT_MODEL) and os.path.exists(UPDATE_MODEL):
            tracker = LightTrackEngine(INIT_MODEL, UPDATE_MODEL, device=DEVICE)

            # --- 预热 (Warm-up) ---
            print(" 正在预热追踪器...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_bbox = [100, 100, 50, 50]
            tracker.init(dummy_frame, dummy_bbox)
            tracker.track(dummy_frame)
            print("✅ 追踪器预热完成！")
            # ---------------------
        else:
            print(f"❌ 错误: 找不到 LightTrack 模型文件")
            return

    # B. 初始化 YOLO
    yolo_detector = None
    if ENABLE_CONFIG["VISUAL_DETECT"]:
        print(f"️ 初始化 YOLO 检测器: {YOLO_ENGINE_PATH}...")
        if os.path.exists(YOLO_ENGINE_PATH):
            # 初始化检测器实例
            yolo_detector = YOLO_Detector(YOLO_ENGINE_PATH, conf_thresh=0.5)

            # ================= YOLO 预热 =================
            print(" 正在预热 YOLO...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            yolo_detector.detect(dummy_img)  # 跑一次空推理
            print("✅ YOLO 预热完成")
            # ============================================
        else:
            print(f"❌ 错误: 找不到 YOLO 权重文件 -> {YOLO_ENGINE_PATH}")
            return  # 无法继续

    # ==================== 2. 主循环 ====================

    ret, prev_frame = cap.read()
    if not ret: return

    # 状态变量
    frame_count = 0
    tracking_state = False  # False=Searching, True=Tracking
    visual_fail_count = 0  # 记录 YOLO 连续失败次数

    # FPS 变量
    prev_frame_time = time.time()
    current_fps = 0.0

    print(f"\n✅ 系统启动. 当前配置: {ENABLE_CONFIG}")

    cv2.namedWindow("System Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("System Demo", 640, 640)

    try:
        while True:
            ret, curr_frame = cap.read()
            if not ret: break

            # === 计算实时 FPS ===
            curr_time = time.time()
            time_delta = curr_time - prev_frame_time
            prev_frame_time = curr_time

            if time_delta > 0:
                instant_fps = 1.0 / time_delta
                current_fps = 0.1 * instant_fps + 0.9 * current_fps
            else:
                current_fps = 0.0
            # ===================

            frame_count += 1
            display_frame = curr_frame.copy()

            # ---------------- 核心逻辑分支 ----------------

            # 分支 1: 搜索模式 (Searching)
            if not tracking_state or not ENABLE_CONFIG["TRACKING"]:

                bbox = None
                search_mode = ""

                # --- 策略 A: 视觉检测 (YOLO) ---
                can_use_visual = ENABLE_CONFIG["VISUAL_DETECT"] and (yolo_detector is not None)
                force_visual = can_use_visual and not ENABLE_CONFIG["MOTION_DETECT"]

                if can_use_visual and (visual_fail_count < VISUAL_FAIL_THRESHOLD or force_visual):
                    search_mode = "VISUAL"
                    # 返回值可能是 None 或 [x, y, w, h, conf]
                    bbox = yolo_detector.detect(curr_frame)

                    if bbox is not None:
                        visual_fail_count = 0
                    else:
                        visual_fail_count += 1

                # --- 策略 B: 运动检测 (MOD) ---
                elif ENABLE_CONFIG["MOTION_DETECT"] and bbox is None:
                    search_mode = "MOTION"
                    detected_boxes = MOD2_global(prev_frame, curr_frame)

                    # 兼容 MOD 可能返回的各种奇怪格式
                    if detected_boxes:
                        if isinstance(detected_boxes, list) and len(detected_boxes) > 0:
                            if isinstance(detected_boxes[0], list) or isinstance(detected_boxes[0], np.ndarray):
                                bbox = detected_boxes[0]  # 取第一个
                            else:
                                bbox = detected_boxes  # 本身就是 [x,y,w,h]
                        elif isinstance(detected_boxes, np.ndarray):
                            bbox = detected_boxes

                # --- 状态转换: 处理 bbox 并初始化追踪器 ---
                if bbox is not None:
                    # ★★★ 关键修复：处理不同来源的 bbox 格式 ★★★
                    clean_bbox = []
                    conf_val = 0.0

                    # 情况 1: YOLO 返回 [x, y, w, h, conf]
                    if len(bbox) == 5:
                        x, y, w, h = map(int, bbox[:4])  # 只取前4个转 int
                        conf_val = bbox[4]
                        clean_bbox = [x, y, w, h]

                    # 情况 2: MOD 返回 [x, y, w, h]
                    else:
                        x, y, w, h = map(int, bbox[:4])
                        clean_bbox = [x, y, w, h]

                    # 绘制搜索到的框
                    color = (255, 0, 0) if search_mode == "VISUAL" else (0, 255, 0)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                    # 标签显示
                    label = f"Init: {search_mode}"
                    if search_mode == "VISUAL":
                        label += f" ({conf_val:.2f})"
                    cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 如果开启了跟踪，则切换状态
                    if ENABLE_CONFIG["TRACKING"]:
                        # ★★★ 传入清洗后的 4 元素 bbox ★★★
                        tracker.init(curr_frame, clean_bbox)
                        tracking_state = True
                        print(f"Frame {frame_count}:  目标锁定 ({search_mode}) -> 切换跟踪")

                else:
                    # 没找到目标，显示搜索状态
                    status_color = (0, 165, 255) if search_mode == "VISUAL" else (0, 0, 255)
                    fail_info = f"{visual_fail_count}/{VISUAL_FAIL_THRESHOLD}" if search_mode == "VISUAL" else ""
                    msg = f"SEARCHING ({search_mode} {fail_info})"
                    cv2.putText(display_frame, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 分支 2: 跟踪模式 (Tracking)
            else:
                # 只有 ENABLE_CONFIG["TRACKING"] 为 True 才会进这里
                bbox, score = tracker.track(curr_frame)

                # 阈值判断 (丢帧检测)
                if score <= 0.98:  # 建议根据实际情况调整，0.98 可能太高了，通常 0.5-0.7
                    print(f"Frame {frame_count}: ⚠️ 追踪丢失 (Score: {score:.2f}) -> 切回搜索")
                    tracking_state = False
                    visual_fail_count = 0  # 给 YOLO 机会

                    # 绘制最后的失败框
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Track: {score:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --------------------------------------------

            prev_frame = curr_frame.copy()

            # UI 绘制
            state_text = "TRACKING" if tracking_state else "SEARCHING"
            state_color = (0, 255, 255) if tracking_state else (0, 0, 255)
            fps_color = (0, 255, 0) if current_fps >= 30 else (0, 255, 255)

            cv2.putText(display_frame, f"Mode: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color,
                        2)

            config_str = f"V:{int(ENABLE_CONFIG['VISUAL_DETECT'])} M:{int(ENABLE_CONFIG['MOTION_DETECT'])} T:{int(ENABLE_CONFIG['TRACKING'])}"
            cv2.putText(display_frame, config_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (254, 0, 0), 1)

            # ==================== 【新增】写入视频帧 ====================
            if video_writer is not None:
                video_writer.write(display_frame)
            # ========================================================

            cv2.imshow("System Demo", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ==================== 【新增】释放资源 ====================
        if video_writer is not None:
            video_writer.release()
            print("\n 视频保存完成。")
        # ========================================================

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()