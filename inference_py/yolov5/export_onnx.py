import torch
import onnx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.experimental import attempt_load
except ImportError:
    print("错误: 找不到 YOLOv5 模块。请确保你在 YOLOv5 源码根目录下运行此脚本。")
    sys.exit(1)

# ---------------- 配置路径 ----------------
weights_path = "/home/verser/Python/Light_DT/python_code/yolov5/yolov5s_GLAD.pt"
onnx_path = "yolov5_drone_op11.onnx"


import torch.nn as nn
def patched_upsample_forward(self, input):
    return torch.nn.functional.interpolate(
        input, self.size, self.scale_factor, self.mode, self.align_corners
    )
nn.Upsample.forward = patched_upsample_forward

# 1. 正确加载 YOLOv5 模型
print("正在加载 YOLOv5 模型...")
device = torch.device('cpu') # 导出 ONNX 用 CPU 即可
# attempt_load 会解析 .pt 文件并重建 PyTorch 模型
model = attempt_load(weights_path, map_location=device)
model.eval()

# 2. 准备输入 (尺寸需和地平线模型输入保持一致)
print("准备虚拟输入...")
dummy_input = torch.randn(1, 3, 640, 640).to(device)

# 3. 导出 ONNX
print(f"正在导出 ONNX 到 {onnx_path} (Opset 11)...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,        # 【关键】强制指定为 11
    input_names=['images'],  # 输入节点名称
    output_names=['output'], # 输出节点名称
    do_constant_folding=True # 开启常量折叠优化
)

# 4. 验证并【顺手解决地平线的 IR 版本限制】
print("验证并处理 IR 版本...")
onnx_model = onnx.load(onnx_path)

print(f"刚导出时的 IR Version: {onnx_model.ir_version}")
print(f"刚导出时的 Opset Version: {onnx_model.opset_import[0].version}")

# 如果 IR 版本大于 9，强制降级 (专为地平线 hb_mapper 准备)
if onnx_model.ir_version > 9:
    print("检测到 IR 版本 > 9，正在强制降级为 9...")
    onnx_model.ir_version = 9
    onnx.save(onnx_model, onnx_path)
    print("IR 版本降级完成并重新保存。")

print("\n--- 最终模型状态 ---")
print(f"最终 IR Version: {onnx_model.ir_version} (应 <= 9)")
print(f"最终 Opset Version: {onnx_model.opset_import[0].version} (应 == 11)")
print("模型已完全准备好，可以扔给 hb_mapper checker 测试了！")