import cv2
import time
import numpy as np
import sys
import os
import ctypes

# ==================== 导入自定义模块 ====================

from MOD2 import MOD2_global
from LightTrack_FastTrack import LightTrackEngine
from YOLO_Engine import YOLO_Detector


# ==================== 核心配置 ====================
# 1. 路径配置
VIDEO_PATH = "/home/verse/Videos/phantom13.mp4"
INIT_MODEL = "./model/ligthtrack_init.pt"
UPDATE_MODEL = "./model/ligthtrack_update.pt"
YOLO_ENGINE_PATH = "./model/DT_Drone.engine"
PLUGIN_LIBRARY = "./model/libmyplugins.so"

# 2. 设备配置
DEVICE = 'cuda'

# 3. 策略阈值
VISUAL_FAIL_THRESHOLD = 120  # YOLO 连续失败多少次切换到 MOD
MOTION_FAIL_THRESHOLD = 120  # MOD 连续失败多少次切回 YOLO

# 4. LightTrack 参数 (用于反算搜索框，必须与 Engine 内部一致)
LT_INSTANCE_SIZE = 288    # 搜索图像大小 (288 or 255)
LT_EXEMPLAR_SIZE = 127    # 模板图像大小
LT_CONTEXT_AMOUNT = 0.5   # 上下文比例

# 5. 功能开关
ENABLE_CONFIG = {
    "VISUAL_DETECT": False,  # 是否开启 YOLO 视觉检测
    "MOTION_DETECT": True,  # 是否开启 MOD2 运动检测
    "TRACKING": True,       # 是否开启 LightTrack 局部跟踪
}

# 加载 TensorRT 插件
if os.path.exists(PLUGIN_LIBRARY):
    try:
        ctypes.CDLL(PLUGIN_LIBRARY)
    except OSError:
        print(f"⚠️ 警告: 无法加载插件库 {PLUGIN_LIBRARY}")
else:
    print(f"⚠️ 警告: 找不到插件库 {PLUGIN_LIBRARY}，YOLO 可能无法运行")

def get_search_bbox(target_bbox, im_shape):
    """
    根据当前目标框计算 LightTrack 的搜索区域框
    """
    x, y, w, h = target_bbox
    # 目标中心
    cx, cy = x + w / 2, y + h / 2

    # 1. 计算包含上下文的目标尺寸 (s_z)
    wc_z = w + LT_CONTEXT_AMOUNT * (w + h)
    hc_z = h + LT_CONTEXT_AMOUNT * (w + h)
    s_z = np.sqrt(wc_z * hc_z)

    # 2. 计算缩放比例 (scale_z)
    scale_z = LT_EXEMPLAR_SIZE / s_z

    # 3. 计算实际搜索区域大小 (s_x)
    s_x = s_z * (LT_INSTANCE_SIZE / LT_EXEMPLAR_SIZE)

    # 4. 生成搜索框坐标 (左上角 x, y, w, h)
    sx = int(cx - s_x / 2)
    sy = int(cy - s_x / 2)
    sw = int(s_x)
    sh = int(s_x)

    return [sx, sy, sw, sh]

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
        if os.path.exists(INIT_MODEL) and os.path.exists(UPDATE_MODEL):
            tracker = LightTrackEngine(INIT_MODEL, UPDATE_MODEL, device=DEVICE)
            # --- 预热 (Warm-up) ---
            print("🔥 正在预热追踪器...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_bbox = [100, 100, 50, 50]
            tracker.init(dummy_frame, dummy_bbox)
            tracker.track(dummy_frame)
            print("✅ 预热完成！")
        else:
            print("❌ LightTrack 模型文件缺失")
            return

    # B. 初始化 YOLO
    yolo_detector = None
    if ENABLE_CONFIG["VISUAL_DETECT"]:
        print(f"👁️ 初始化 YOLO 检测器: {YOLO_ENGINE_PATH}...")
        if os.path.exists(YOLO_ENGINE_PATH):
            yolo_detector = YOLO_Detector(YOLO_ENGINE_PATH)
            print("🔥 正在预热 YOLO...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            yolo_detector.detect(dummy_img)
            print("✅ YOLO 预热完成")
        else:
            print(f"❌ 错误: 找不到 YOLO 引擎文件 -> {YOLO_ENGINE_PATH}")
            return

    # ==================== 2. 主循环 ====================

    ret, prev_frame = cap.read()
    if not ret: return

    # 状态变量
    frame_count = 0
    tracking_state = False

    # 计数器
    visual_fail_count = 0
    motion_fail_count = 0  # [新增] 初始化运动检测失败计数器

    # FPS 变量
    prev_frame_time = time.time()
    current_fps = 0.0

    print(f"\n✅ 系统启动. 当前配置: {ENABLE_CONFIG}")
    print(f"策略: YOLO失败 {VISUAL_FAIL_THRESHOLD} 次 -> MOD; MOD失败 {MOTION_FAIL_THRESHOLD} 次 -> YOLO")

    cv2.namedWindow("System Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("System Demo", 640, 640)

    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        # FPS 计算
        curr_time = time.time()
        time_delta = curr_time - prev_frame_time
        prev_frame_time = curr_time

        if time_delta > 0:
            instant_fps = 1.0 / time_delta
            current_fps = 0.1 * instant_fps + 0.9 * current_fps
        else:
            current_fps = 0.0

        frame_count += 1
        display_frame = curr_frame.copy()

        # ---------------- 核心逻辑分支 ----------------
        # 分支 1: 搜索模式 (Searching)
        if not tracking_state or not ENABLE_CONFIG["TRACKING"]:
            bbox = None
            search_mode = ""

            # --- YOLO 逻辑 ---
            # 开启 YOLO 且 (失败次数未超标 或 强制使用 或 没有开MOD)
            can_use_visual = ENABLE_CONFIG["VISUAL_DETECT"] and (yolo_detector is not None)
            force_visual = can_use_visual and not ENABLE_CONFIG["MOTION_DETECT"]

            if can_use_visual and (visual_fail_count < VISUAL_FAIL_THRESHOLD or force_visual):
                search_mode = "VISUAL"
                bbox = yolo_detector.detect(curr_frame)

                if bbox is not None:
                    visual_fail_count = 0
                    motion_fail_count = 0 # [新增] YOLO 成功，重置 MOD 计数
                else:
                    visual_fail_count += 1

            # --- MOD (运动检测) 逻辑 ---
            elif ENABLE_CONFIG["MOTION_DETECT"] and bbox is None:
                search_mode = "MOTION"
                # 传入 copy 以防修改原图影响后续处理
                detected_boxes = MOD2_global(prev_frame.copy(), curr_frame.copy())

                # --- 🛠️ 修复开始: 增加对 Tuple 的兼容性 ---
                if detected_boxes:
                    # 情况 A: 返回的是 Tuple (x, y, w, h) -> 你的 MOD2 代码就是这种情况
                    if isinstance(detected_boxes, tuple) and len(detected_boxes) == 4:
                        bbox = list(detected_boxes) # 转为 list 以保持一致性

                    # 情况 B: 返回的是 List (例如 [[x,y,w,h]])
                    elif isinstance(detected_boxes, list) and len(detected_boxes) > 0:
                        # 如果是嵌套列表 [[x,y,w,h]]
                        if isinstance(detected_boxes[0], (list, tuple, np.ndarray)):
                            bbox = list(detected_boxes[0])
                        # 如果是扁平列表 [x,y,w,h]
                        else:
                            bbox = detected_boxes

                    # 情况 C: 返回的是 Numpy Array
                    elif isinstance(detected_boxes, np.ndarray):
                        bbox = detected_boxes.tolist()

            # --- Init Tracker (初始化追踪) ---
            if bbox is not None:
                x, y, w, h = map(int, bbox)
                color = (255, 0, 0) if search_mode == "VISUAL" else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"Init: {search_mode}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if ENABLE_CONFIG["TRACKING"]:
                    tracker.init(curr_frame, bbox)
                    tracking_state = True
                    # [关键] 进入跟踪状态，所有搜索计数归零
                    visual_fail_count = 0
                    motion_fail_count = 0
                    print(f"Frame {frame_count}: 🎯 目标锁定 ({search_mode}) -> 切换跟踪")
            else:
                # 显示搜索状态文本
                status_color = (0, 165, 255) if search_mode == "VISUAL" else (0, 0, 255)

                if search_mode == "VISUAL":
                    fail_info = f"{visual_fail_count}/{VISUAL_FAIL_THRESHOLD}"
                else:
                    fail_info = f"{motion_fail_count}/{MOTION_FAIL_THRESHOLD}" # 显示 MOD 计数

                msg = f"SEARCHING ({search_mode} {fail_info})"
                cv2.putText(display_frame, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 分支 2: 跟踪模式 (Tracking)
        else:
            # 1. 执行跟踪
            bbox, score = tracker.track(curr_frame)

            # 2. 计算并绘制 LightTrack 的搜索区域
            sx, sy, sw, sh = get_search_bbox(bbox, curr_frame.shape)
            cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)
            cv2.putText(display_frame, "Search Area", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 3. 结果判断
            if score < 0.98:
                print(f"Frame {frame_count}: ⚠️ 追踪丢失 (Score: {score:.2f}) -> 切回搜索")
                tracking_state = False

                # 丢失时，重置搜索计数，让系统按优先级 (YOLO -> MOD) 重新开始
                visual_fail_count = 0
                motion_fail_count = 0

                x, y, w, h = map(int, bbox)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display_frame, f"Track: {score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --------------------------------------------

        prev_frame = curr_frame.copy()

        # UI Info
        state_text = "TRACKING" if tracking_state else "SEARCHING"
        state_color = (0, 255, 255) if tracking_state else (0, 0, 255)
        fps_color = (0, 255, 0) if current_fps >= 30 else (0, 255, 255)

        cv2.putText(display_frame, f"Mode: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

        config_str = f"V:{int(ENABLE_CONFIG['VISUAL_DETECT'])} M:{int(ENABLE_CONFIG['MOTION_DETECT'])} T:{int(ENABLE_CONFIG['TRACKING'])}"
        cv2.putText(display_frame, config_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (254, 0, 0), 1)

        cv2.imshow("System Demo", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()