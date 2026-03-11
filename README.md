# 🚀 Hybrid High-Performance Object Tracking System

基于 **YOLOv5 (TensorRT)**、**LightTrack** 和 **MOD (运动目标检测)** 的混合目标检测跟踪系统。

[//]: # (<div align="center">)

[//]: # (  <img src="assets/result_.png" width="80%" alt="小目标无人机检测框架" />)

[//]: # (</div>)

<div align="center">
  <a href="assets/result.mp4.mp4">
    <img src="assets/result_.png" width="80%" alt="点击观看视频">
  </a>
  <p>👆 点击图片播放视频</p>
</div>

## ✨ 核心特性

* **⚡ TensorRT 加速**: YOLOv5 检测器经过 TensorRT 引擎量化与加速，支持 FP16/FP32 推理。
* **🛠️ 混合架构 (TBD)**:
    * **检测 (Detect)**: 结合 YOLOv5 (针对已知类别) 和 MOD (针对运动物体) 进行全局搜索。
    * **跟踪 (Track)**: 使用 LightTrack 进行高帧率、高精度的单目标持续跟踪。
## 🏗️ 系统架构

<div align="center">
  <img src="assets/小目标无人机检测框架.png" width="80%" alt="小目标无人机检测框架" />
</div>

系统采用有限状态机 (FSM) 进行调度：
1.  **SEARCHING (搜索模式)**:
    * 优先使用 **YOLOv5** 检测目标。
    * 若 YOLO 连续 N 帧失败，自动降级为 **MOD (运动检测)**。
2.  **INITIALIZING (初始化过渡)**:
    * 发现目标后的下一帧，利用缓存数据初始化 LightTrack，同时对当前帧进行跟踪（对用户透明，无感知延迟）。
3.  **TRACKING (跟踪模式)**:
    * 使用 **LightTrack** 进行高速跟踪。
    * 实时监控置信度 (Score)，若低于阈值 (如 0.98) 则触发重检测机制。
