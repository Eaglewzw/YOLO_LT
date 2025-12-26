import cv2
import time
import numpy as np
import sys
import os
import ctypes

# ==================== 导入自定义模块 ====================
from MOD2 import MOD2_global
from LightTrack_Engine import LightTrackEngine
from YOLO_Engine import YOLO_Detector

# ==================== 核心配置 ====================
# 1. 路径配置
VIDEO_PATH = "/home/verse/Videos/phantom13.mp4"
INIT_MODEL = "./model/ligthtrack_init.pt"
UPDATE_MODEL = "./model/ligthtrack_update.pt"
YOLO_ENGINE_PATH = "./model/yolov5s_GLAD_wzw.engine"
PLUGIN_LIBRARY = "./model/libmyplugins.so"

# 2. 设备配置
DEVICE = 'cuda'

# 3. 策略阈值
VISUAL_FAIL_THRESHOLD = 120

# 4. 功能开关
ENABLE_CONFIG = {
    "VISUAL_DETECT": True,  # 是否开启 YOLO 视觉检测
    "MOTION_DETECT": True,  # 是否开启 MOD2 运动检测
    "TRACKING": True,       # 是否开启 LightTrack 局部跟踪
}

# 加载 TensorRT 插件
if os.path.exists(PLUGIN_LIBRARY):
    ctypes.CDLL(PLUGIN_LIBRARY)
else:
    print(f"⚠️ 警告: 找不到插件库 {PLUGIN_LIBRARY}，YOLO 可能无法运行")

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return

    # ==================== 1. 模型初始化 (循环外) ====================

    # A. 初始化 LightTrack
    tracker = None
    if ENABLE_CONFIG["TRACKING"]:
        print("🚀 初始化 LightTrack 追踪器...")
        tracker = LightTrackEngine(INIT_MODEL, UPDATE_MODEL, device=DEVICE)

        # --- 预热 (Warm-up) ---
        print("🔥 正在预热追踪器...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 100, 50, 50]
        tracker.init(dummy_frame, dummy_bbox) # 统一使用 init
        tracker.track(dummy_frame)
        print("✅ 预热完成！")
        # ---------------------

    # B. 初始化 YOLO (⚠️ 关键修正：必须在循环外初始化)
    yolo_detector = None
    if ENABLE_CONFIG["VISUAL_DETECT"]:
        print(f"👁️ 初始化 YOLO 检测器: {YOLO_ENGINE_PATH}...")
        if os.path.exists(YOLO_ENGINE_PATH):
            yolo_detector = YOLO_Detector(YOLO_ENGINE_PATH)
            print("✅ YOLO 加载成功")
        else:
            print(f"❌ 错误: 找不到 YOLO 引擎文件 -> {YOLO_ENGINE_PATH}")
            return # 无法继续

    # ==================== 2. 主循环 ====================

    ret, prev_frame = cap.read()
    if not ret: return

    # 状态变量
    frame_count = 0
    tracking_state = False # False=Searching, True=Tracking
    visual_fail_count = 0  # 记录 YOLO 连续失败次数

    # FPS 统计
    fps_start_time = time.time()
    fps_frame_counter = 0
    current_fps = 0.0
    print(f"\n✅ 系统启动. 当前配置: {ENABLE_CONFIG}")

    cv2.namedWindow("System Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("System Demo", 640, 640)

    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        frame_count += 1
        display_frame = curr_frame.copy()

        # ---------------- 核心逻辑分支 ----------------

        # 分支 1: 搜索模式 (Searching)
        # 进入条件: 当前没在跟踪，或者跟踪功能被强制关闭
        if not tracking_state or not ENABLE_CONFIG["TRACKING"]:

            bbox = None
            search_mode = ""

            # --- 策略 A: 视觉检测 (YOLO) ---
            # 启用条件: 开关开启 且 (YOLO还没失败太久 或 MOD被禁用)
            can_use_visual = ENABLE_CONFIG["VISUAL_DETECT"] and (yolo_detector is not None)
            force_visual = can_use_visual and not ENABLE_CONFIG["MOTION_DETECT"]

            if can_use_visual and (visual_fail_count < VISUAL_FAIL_THRESHOLD or force_visual):
                search_mode = "VISUAL"

                # ⚠️ 修正：调用实例的 detect 方法，而不是实例化类
                bbox = yolo_detector.detect(curr_frame)

                if bbox is not None:
                    visual_fail_count = 0 # 成功了，重置失败计数
                else:
                    visual_fail_count += 1 # 失败了，计数+1

            # --- 策略 B: 运动检测 (MOD) ---
            # 启用条件: 开关开启 且 (YOLO失败次数过多 或 YOLO没开) 且 前面没检测到
            elif ENABLE_CONFIG["MOTION_DETECT"] and bbox is None:
                search_mode = "MOTION"
                # 注意：MOD2_global 需要两帧
                detected_boxes = MOD2_global(prev_frame, curr_frame)

                if detected_boxes:
                    # 格式兼容处理
                    if isinstance(detected_boxes, list) and len(detected_boxes) > 0:
                        # 如果返回的是列表的列表 [[x,y,w,h], ...]
                        if isinstance(detected_boxes[0], list) or isinstance(detected_boxes[0], np.ndarray):
                            bbox = detected_boxes[0]
                        else:
                            # 如果返回的是单列表 [x,y,w,h]
                            bbox = detected_boxes
                    elif isinstance(detected_boxes, np.ndarray):
                        bbox = detected_boxes

            # --- 状态转换: 初始化追踪器 ---
            if bbox is not None:
                x, y, w, h = map(int, bbox)

                # 绘制搜索到的框
                color = (255, 0, 0) if search_mode == "VISUAL" else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"Init: {search_mode}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 如果开启了跟踪，则切换状态
                if ENABLE_CONFIG["TRACKING"]:
                    tracker.init(curr_frame, bbox) # 初始化 LightTrack
                    tracking_state = True          # 切换为跟踪状态
                    print(f"Frame {frame_count}: 🎯 目标锁定 ({search_mode}) -> 切换跟踪")

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
            if score < 0.98: # 或者 0.6，看你的 LightTrack 表现
                print(f"Frame {frame_count}: ⚠️ 追踪丢失 (Score: {score:.2f}) -> 切回搜索")
                tracking_state = False # 切回搜索模式
                visual_fail_count = 0  # 重置计数，给 YOLO 机会

                # 绘制最后的失败框
                x, y, w, h = map(int, bbox)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display_frame, f"LightTrack: {score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --------------------------------------------

        # FPS 计算
        fps_frame_counter += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 0.5:
            current_fps = fps_frame_counter / elapsed_time
            fps_frame_counter = 0
            fps_start_time = time.time()

        prev_frame = curr_frame.copy()

        # UI 绘制
        state_text = "TRACKING" if tracking_state else "SEARCHING"
        state_color = (0, 255, 255) if tracking_state else (0, 0, 255)
        fps_color = (0, 255, 0) if current_fps >= 30 else (0, 255, 255)

        cv2.putText(display_frame, f"Mode: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

        # 显示开关状态
        config_str = f"V:{int(ENABLE_CONFIG['VISUAL_DETECT'])} M:{int(ENABLE_CONFIG['MOTION_DETECT'])} T:{int(ENABLE_CONFIG['TRACKING'])}"
        cv2.putText(display_frame, config_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (254, 0, 0), 1)

        cv2.imshow("System Demo", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()