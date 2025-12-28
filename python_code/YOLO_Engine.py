# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Optimized YOLO Detector with TensorRT
"""

import os
import time
import ctypes
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

# 加载自定义插件 (如果存在)
PLUGIN_LIBRARY = "./model/libmyplugins.so"
if os.path.exists(PLUGIN_LIBRARY):
    ctypes.CDLL(PLUGIN_LIBRARY)
else:
    print(f"⚠️ 警告: 找不到插件库 {PLUGIN_LIBRARY}，YOLO 可能无法运行")


class YOLO_Detector:
    def __init__(self, engine_file_path, input_shape=(640, 640), conf_thresh=0.5, iou_thresh=0.4):
        self.input_w, self.input_h = input_shape
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # 1. 初始化 CUDA 上下文
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()

        try:
            # 2. 加载 TensorRT 引擎
            TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(TRT_LOGGER)

            if not os.path.exists(engine_file_path):
                raise FileNotFoundError(f"Engine file not found: {engine_file_path}")

            with open(engine_file_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            # 3. 分配内存 (Host & Device)
            self.host_inputs, self.cuda_inputs = [], []
            self.host_outputs, self.cuda_outputs = [], []
            self.bindings = []

            for i in range(self.engine.num_bindings):
                tensor_name = self.engine.get_binding_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

                # 计算 Buffer 大小
                size = 1
                for s in shape:
                    size *= s if s > 0 else 1  # 处理动态 Batch，默认为 1

                # 分配内存
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)

                self.bindings.append(int(cuda_mem))

                # 区分输入输出
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.host_inputs.append(host_mem)
                    self.cuda_inputs.append(cuda_mem)
                else:
                    self.host_outputs.append(host_mem)
                    self.cuda_outputs.append(cuda_mem)

        finally:
            self.cfx.pop()

    def detect(self, image_raw):
        """执行推理的主函数"""
        self.cfx.push()
        try:
            # 1. 预处理
            input_image, origin_h, origin_w = self.preprocess_image(image_raw)

            # 2. Host -> Device (Async)
            np.copyto(self.host_inputs[0], input_image.ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

            # 3. 执行推理 (自动兼容 Implicit/Explicit Batch)
            if self.engine.has_implicit_batch_dimension:
                self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # 4. Device -> Host (Async)
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)

            # 5. 同步等待
            self.stream.synchronize()

        finally:
            self.cfx.pop()

        # 6. 后处理 (CPU)
        return self.post_process(self.host_outputs[0], origin_h, origin_w)

    def preprocess_image(self, image_raw):
        """图像缩放与填充"""
        h, w, _ = image_raw.shape
        r = min(self.input_h / h, self.input_w / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))

        # 计算 Padding
        dw, dh = self.input_w - new_unpad[0], self.input_h - new_unpad[1]
        dw, dh = dw / 2, dh / 2

        # Resize
        if (w, h) != new_unpad:
            image = cv2.resize(image_raw, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            image = image_raw

        # Add Border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))

        # HWC -> CHW, Normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return np.ascontiguousarray(image), h, w

    def post_process(self, output, origin_h, origin_w):
        """解析推理结果"""
        num = int(output[0])
        if num == 0: return None

        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        boxes = pred[:, :4]
        scores = pred[:, 4]

        # 阈值过滤
        mask = scores > self.conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        if len(boxes) == 0: return None

        # 坐标还原与 NMS
        boxes_xywh = self.scale_coords(self.input_h, self.input_w, boxes, origin_h, origin_w)
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), self.conf_thresh, self.iou_thresh)
        if len(indices) == 0: return None

        # 寻找最高分目标
        best_idx = indices.flatten()[np.argmax(scores[indices.flatten()])]
        return boxes_xywh[best_idx].astype(int)

    def scale_coords(self, img1_h, img1_w, coords, img0_h, img0_w):
        """将预测框坐标从 640x640 映射回原图尺寸"""
        gain = min(img1_h / img0_h, img1_w / img0_w)
        pad_x = (img1_w - img0_w * gain) / 2
        pad_y = (img1_h - img0_h * gain) / 2

        coords[:, 0] -= pad_x
        coords[:, 1] -= pad_y
        coords[:, :4] /= gain
        coords[:, 0] -= coords[:, 2] / 2
        coords[:, 1] -= coords[:, 3] / 2
        return coords

    def destroy(self):
        try:
            self.cfx.pop()
        except:
            pass


def main():
    # ================= 配置 =================
    # engine_path = "./model/yolov5s_GLAD_wzw.engine"
    engine_path = "./model/DT_Drone.engine"
    video_path = "/home/verse/Videos/fast_drone.mp4"
    # =======================================

    detector = YOLO_Detector(engine_path)


    cv2.namedWindow("TensorRT YOLO Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TensorRT YOLO Detector", 640, 640)

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

            fps = 1.0 / (time.time() - t_start)

            # 绘制结果
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Target | FPS: {fps:.1f}"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"FPS: {fps:.1f} (No Target)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("TensorRT YOLO Detector", frame)
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