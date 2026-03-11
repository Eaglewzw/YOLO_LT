import cv2
import time
import numpy as np
import sys
import os
import ctypes
from datetime import datetime



# ================= 配置开关 =================
USE_TENSORRT_10 = True  # True: 使用 TRT10, False: 使用 TRT8.6
# ===========================================
INIT_MODEL = "./model/ligthtrack_init.pt"
UPDATE_MODEL = "./model/ligthtrack_update.pt"


# 导入 LightTrack
from LightTrack_FastTrack import LightTrackEngine
from MOD2 import MOD2_global

# ================= 1. 动态导入 YOLO 模块 =================
if USE_TENSORRT_10:
    print("正在加载 TensorRT 10 模块...")
    from modules_trt10.YOLO_Engine_TRT10 import YOLO_Detector

    # TRT 10 路径配置
    YOLO_ENGINE_PATH = "./model/TensorRT_10/GDUT_UAV.pt"
    PLUGIN_LIBRARY = "./model/TensorRT_10/libmyplugins.so"
else:
    print("正在加载 TensorRT 8.6 模块...")
    from YOLO_Engine import YOLO_Detector

    # TRT 8.6 路径配置
    YOLO_ENGINE_PATH = "./model/TensorRT_8/yolov5s_GLAD_wzw.engine"
    PLUGIN_LIBRARY = "./model/TensorRT_8/libmyplugins.so"

    # 加载旧版插件
    if os.path.exists(PLUGIN_LIBRARY):
        try:
            ctypes.CDLL(PLUGIN_LIBRARY)
        except OSError:
            print(f"⚠️ 警告: 无法加载插件 {PLUGIN_LIBRARY}")
    else:
        print(f"⚠️ 警告: 找不到插件库 {PLUGIN_LIBRARY}")

# ==================== 2. 核心配置 ====================
VIDEO_PATH = "/home/verser/Videos/fast_drone.mp4"
DEVICE = 'cuda'

# --- 策略阈值 (新增) ---
VISUAL_FAIL_THRESHOLD = 30  # YOLO 连续失败 5 帧 -> 切换 MOD
MOD_FAIL_THRESHOLD = 10  # MOD 连续失败 10 帧 -> 重置回 YOLO

# --- LightTrack 可视化参数 (新增) ---
LT_INSTANCE_SIZE = 288  # 搜索图像大小
LT_EXEMPLAR_SIZE = 127  # 模板图像大小
LT_CONTEXT_AMOUNT = 0.5  # 上下文比例

ENABLE_CONFIG = {
    "VISUAL_DETECT": True,  # 是否开启 YOLO 视觉检测
    "MOTION_DETECT": True,  # 是否开启 MOD2 运动检测
    "TRACKING": True,  # 是否开启 LightTrack 局部跟踪
}


# ==================== 3. 辅助函数 ====================
def get_search_bbox(target_bbox, im_shape):
    """
    根据当前目标框计算 LightTrack 的搜索区域框 (用于可视化)
    """
    if target_bbox is None: return [0, 0, 0, 0]

    # 【安全处理】只取前4个值 [x, y, w, h]
    x, y, w, h = target_bbox[:4]
    cx, cy = x + w / 2, y + h / 2

    # 1. 计算包含上下文的目标尺寸
    wc_z = w + LT_CONTEXT_AMOUNT * (w + h)
    hc_z = h + LT_CONTEXT_AMOUNT * (w + h)
    s_z = np.sqrt(wc_z * hc_z)

    # 2. 计算缩放比例
    scale_z = LT_EXEMPLAR_SIZE / s_z

    # 3. 计算实际搜索区域大小
    s_x = s_z * (LT_INSTANCE_SIZE / LT_EXEMPLAR_SIZE)

    # 4. 生成坐标
    sx = int(cx - s_x / 2)
    sy = int(cy - s_x / 2)
    sw = int(s_x)
    sh = int(s_x)

    return [sx, sy, sw, sh]


def get_safe_roi(frame, bbox, expansion_factor=1.5):
    """
    安全裁剪 ROI 区域，并向外扩张一定比例
    :param frame: 原图
    :param bbox: [x, y, w, h]
    :param expansion_factor: 扩张系数，1.5 表示宽高扩大为原来的 1.5 倍
    :return: 裁剪后的图像片段, 裁剪区域的坐标 (x1, y1, x2, y2)
    """
    if bbox is None: return None, None

    img_h, img_w = frame.shape[:2]
    x, y, w, h = map(int, bbox[:4])

    # 计算中心点
    cx, cy = x + w / 2, y + h / 2

    # 计算新的宽高
    new_w = w * expansion_factor
    new_h = h * expansion_factor

    # 计算新的左上角和右下角坐标
    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(img_w, cx + new_w / 2))
    y2 = int(min(img_h, cy + new_h / 2))

    # 裁剪
    roi = frame[y1:y2, x1:x2]

    # 如果裁剪区域太小（比如飞出屏幕），返回 None
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return None, None

    return roi, (x1, y1, x2, y2)

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return

    output_dir = "../assets"
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

    # ==================== 4. 模型初始化 ====================
    # A. 初始化 LightTrack
    tracker = None
    if ENABLE_CONFIG["TRACKING"]:
        print(" 初始化 LightTrack 追踪器...")
        if os.path.exists(INIT_MODEL) and os.path.exists(UPDATE_MODEL):
            tracker = LightTrackEngine(INIT_MODEL, UPDATE_MODEL, device=DEVICE)
            # --- 预热 ---
            print(" 正在预热追踪器...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_bbox = [100, 100, 50, 50]
            tracker.init(dummy_frame, dummy_bbox)
            tracker.track(dummy_frame)
            print("✅ 预热完成！")
        else:
            print(f"❌ 错误: LightTrack 模型不存在 -> {INIT_MODEL}")
            return

    # B. 初始化 YOLO
    yolo_detector = None
    if ENABLE_CONFIG["VISUAL_DETECT"]:
        print(f"️ 初始化 YOLO 检测器: {YOLO_ENGINE_PATH}...")
        if os.path.exists(YOLO_ENGINE_PATH):
            yolo_detector = YOLO_Detector(YOLO_ENGINE_PATH)
            # --- 预热 ---
            print(" 正在预热 YOLO...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            yolo_detector.detect(dummy_img)
            print("✅ YOLO 预热完成")
        else:
            print(f"❌ 错误: 找不到 YOLO 引擎文件")
            return

    # ==================== 5. 主循环 ====================
    ret, prev_frame = cap.read()
    if not ret: return

    frame_count = 0
    tracking_state = False

    # 计数器
    visual_fail_count = 0  # YOLO 连续失败次数
    mod_fail_count = 0  # MOD 连续失败次数

    prev_frame_time = time.time()
    current_fps = 0.0

    print(f"\n✅ 系统启动. 当前配置: {ENABLE_CONFIG}")
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

            # --- 策略 A: 视觉检测 (YOLO) ---
            can_use_visual = ENABLE_CONFIG["VISUAL_DETECT"] and (yolo_detector is not None)
            force_visual = can_use_visual and not ENABLE_CONFIG["MOTION_DETECT"]

            # 逻辑：如果 YOLO 失败次数未超标，或者被强制使用(没开MOD)
            if can_use_visual and (visual_fail_count < VISUAL_FAIL_THRESHOLD or force_visual):
                search_mode = "VISUAL"
                bbox = yolo_detector.detect(curr_frame)

                if bbox is not None:
                    visual_fail_count = 0
                    mod_fail_count = 0  # 只要检测到，重置所有计数
                else:
                    visual_fail_count += 1

            # --- 策略 B: 运动检测 (MOD) ---
            # 逻辑：开启了 MOD 且 (YOLO 没开 或 YOLO 失败次数过多)
            elif ENABLE_CONFIG["MOTION_DETECT"] and bbox is None:
                search_mode = "MOTION"
                detected_boxes = MOD2_global(prev_frame, curr_frame)

                # MOD 返回值兼容处理 (Tuple/List/Array)
                if detected_boxes:
                    if isinstance(detected_boxes, tuple) and len(detected_boxes) == 4:
                        bbox = list(detected_boxes)
                    elif isinstance(detected_boxes, list) and len(detected_boxes) > 0:
                        bbox = detected_boxes[0] if isinstance(detected_boxes[0],
                                                               (list, np.ndarray)) else detected_boxes
                    elif isinstance(detected_boxes, np.ndarray):
                        bbox = detected_boxes.tolist()

                    mod_fail_count = 0  # 成功重置 MOD 计数
                else:
                    # MOD 失败处理
                    mod_fail_count += 1
                    # 【新增策略】如果 MOD 也连续失败，重置系统回 YOLO 状态
                    if mod_fail_count >= MOD_FAIL_THRESHOLD:
                        print(f"Frame {frame_count}: ⚠️ MOD 连续失败 {MOD_FAIL_THRESHOLD} 帧 -> 重置回 YOLO")
                        visual_fail_count = 0
                        mod_fail_count = 0

            # --- 状态转换: 初始化追踪器 ---
            if bbox is not None:
                # 【安全处理】只取前4个值
                x, y, w, h = map(int, bbox[:4])

                # 绘制 Init 框
                # color = (255, 0, 0) if search_mode == "VISUAL" else (0, 255, 0)
                # cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                label = f"Init: {search_mode}"
                # 如果是 YOLO 且有置信度，显示置信度
                if search_mode == "VISUAL" and len(bbox) >= 5:
                    label += f" ({bbox[4]:.2f})"
                # cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if ENABLE_CONFIG["TRACKING"]:
                    tracker.init(curr_frame, bbox[:4])  # 传入清洗后的 bbox
                    tracking_state = True
                    # 进入跟踪后，重置搜索计数
                    visual_fail_count = 0
                    mod_fail_count = 0
                    print(f"Frame {frame_count}:  目标锁定 ({search_mode}) -> 切换跟踪")
            else:
                # 显示搜索状态文本
                status_color = (0, 165, 255) if search_mode == "VISUAL" else (0, 0, 255)
                # 显示当前模式的失败计数
                fail_info = f"{visual_fail_count}/{VISUAL_FAIL_THRESHOLD}" if search_mode == "VISUAL" else f"{mod_fail_count}/{MOD_FAIL_THRESHOLD}"
                msg = f"SEARCHING ({search_mode} {fail_info})"
                cv2.putText(display_frame, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 分支 2: 跟踪模式 (Tracking)
        else:
            bbox, score = tracker.track(curr_frame)
            # ================== 绘制 LightTrack 搜索区域 (白色框) ==================
            sx, sy, sw, sh = get_search_bbox(bbox[:4], curr_frame.shape)
            cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)
            cv2.putText(display_frame, "Search Area", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # ==============================================================================

            # ================== 每 120 帧进行一次 YOLO 验证 ==================
            # 只有当开启了视觉检测，且当前处于跟踪状态时执行
            if ENABLE_CONFIG["VISUAL_DETECT"] and frame_count % 30 == 0:

                # 1. 裁剪 ROI (扩大 1.5 倍，防止目标正好在边缘被切断)
                roi_img, roi_coords = get_safe_roi(curr_frame, bbox, expansion_factor=3)

                if roi_img is not None:
                    # 2. 送入 YOLO 检测
                    # 注意：YOLO 可能会把 ROI resize 到 640x640，这正是“局部放大”的效果
                    roi_bbox = yolo_detector.detect(roi_img)

                    # 3. 严格阈值判决
                    # 默认认为验证失败，除非证据确凿
                    is_verified = False
                    verify_score = 0.0

                    if roi_bbox is not None:
                        # ★★★ 提取你的 detect 返回的 conf ★★★
                        verify_score = roi_bbox[4]

                        # ★★★ 关键点：这里用 0.6 或 0.7，比 detect 内部的 0.25 要高 ★★★
                        if verify_score > 0.45:
                            is_verified = True
                        else:
                            print(f"Frame {frame_count}: ⚠️ 验证置信度不足 ({verify_score:.2f} < 0.6)")

                    # 4. 根据结果执行动作
                    if not is_verified:
                        print(f"Frame {frame_count}: ❌ YOLO 验证失败 -> 强制中断跟踪")

                        # 强制中断跟踪
                        tracking_state = False
                        score = 0.0

                        # (可选) 画红框可视化
                        # rx1, ry1, rx2, ry2 = roi_coords
                        # cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                        # cv2.putText(display_frame, f"Fail:{verify_score:.2f}", (rx1, ry1 - 5),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # 验证成功
                        # (可选) 你甚至可以用 YOLO 的框修正 Tracker 的框 (这里暂时不做，仅作验证)
                        print(f"Frame {frame_count}: ✅ YOLO 验证通过")

                        # 可视化验证成功的区域 (紫色框)
                        # rx1, ry1, rx2, ry2 = roi_coords
                        # cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
                        # cv2.putText(display_frame, "Verify OK", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        #             (255, 0, 255), 2)
            # ======================================================================

            # 跟踪结果判断 (原有的 score 判断逻辑)
            # 如果上面把 tracking_state 设为 False 了，这里会直接进入失败处理
            if not tracking_state or score <= 0.98:  # 修改：增加对 tracking_state 的检查
                if tracking_state:  # 只有当是因为 score 低而失败时才打印这句
                    print(f"Frame {frame_count}: ⚠️ 追踪丢失 (Score: {score:.2f}) -> 切回搜索")

                tracking_state = False
                visual_fail_count = 0
                mod_fail_count = 0

                x, y, w, h = map(int, bbox[:4])
                # cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                x, y, w, h = map(int, bbox[:4])
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display_frame, f"Track: {score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # else:
        #     bbox, score = tracker.track(curr_frame)
        #
        #     # 【新增功能】绘制 LightTrack 搜索区域 (白色框)
        #     # 必须使用 bbox[:4] 避免解包错误
        #     sx, sy, sw, sh = get_search_bbox(bbox[:4], curr_frame.shape)
        #     cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)
        #     cv2.putText(display_frame, "Search Area", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #
        #     # 跟踪结果判断
        #     if score <= 0.98:
        #         print(f"Frame {frame_count}: ⚠️ 追踪丢失 (Score: {score:.2f}) -> 切回搜索")
        #         tracking_state = False
        #         visual_fail_count = 0  # 立即允许 YOLO 尝试
        #         mod_fail_count = 0
        #
        #         x, y, w, h = map(int, bbox[:4])
        #         cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     else:
        #         x, y, w, h = map(int, bbox[:4])
        #         cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        #         cv2.putText(display_frame, f"Track: {score:.2f}", (x, y - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --------------------------------------------
        prev_frame = curr_frame.copy()

        # UI Info
        state_text = "TRACKING" if tracking_state else "SEARCHING"
        state_color = (0, 255, 255) if tracking_state else (0, 0, 255)
        fps_color = (0, 255, 0) if current_fps >= 30 else (0, 255, 255)

        cv2.putText(display_frame, f"Mode: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(display_frame, f"FPS: {current_fps * 2 :.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
        config_str = f"V:{int(ENABLE_CONFIG['VISUAL_DETECT'])} M:{int(ENABLE_CONFIG['MOTION_DETECT'])} T:{int(ENABLE_CONFIG['TRACKING'])}"
        cv2.putText(display_frame, config_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (254, 0, 0), 1)

        # ==================== 写入视频帧 ====================
        if video_writer is not None:
            video_writer.write(display_frame)

        cv2.imshow("System Demo", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()