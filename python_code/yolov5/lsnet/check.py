import torch
import json
import os

# ================= 配置路径 =================
# 请修改为你下载文件的实际路径
config_path = './lsnet_b/config.json'
bin_path = './lsnet_b/pytorch_model.bin'


# ===========================================

def inspect_config(json_file):
    print(f"\n{'=' * 20} 1. Inspecting Config ({os.path.basename(json_file)}) {'=' * 20}")

    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        return

    with open(json_file, 'r') as f:
        cfg = json.load(f)

    # 打印一些关键参数 (LSNet 特有的)
    # 这些键名取决于 config.json 的具体内容，通常包含 architecture, hidden_size 等
    print("--- Key Architecture Parameters ---")
    keys_to_check = ['architectures', 'model_type', 'num_classes', 'image_size',
                     'embed_dim', 'depth', 'num_heads', 'hidden_size']

    for k in keys_to_check:
        if k in cfg:
            print(f"{k:<20}: {cfg[k]}")

    # 打印完整内容（如果不想看太长可以注释掉）
    print("\n--- Full Config Dump ---")
    print(json.dumps(cfg, indent=2))


def inspect_weights(bin_file):
    print(f"\n{'=' * 20} 2. Inspecting Weights ({os.path.basename(bin_file)}) {'=' * 20}")

    if not os.path.exists(bin_file):
        print(f"Error: File not found: {bin_file}")
        return

    try:
        # 加载权重字典
        # map_location='cpu' 保证即使没有 GPU 也能查看
        state_dict = torch.load(bin_file, map_location='cpu')
    except Exception as e:
        print(f"Error loading .bin file: {e}")
        return

    print(f"Total number of keys (layers): {len(state_dict)}")

    print("\n--- Layer Analysis (First 20 Layers) ---")
    print(f"{'Key Name':<50} | {'Shape'}")
    print("-" * 70)

    # 遍历打印前 20 层，看看结构
    for i, (key, value) in enumerate(state_dict.items()):
        if i >= 20:
            print("... (remaining layers omitted) ...")
            break
        print(f"{key:<50} | {list(value.shape)}")

    # 检查关键的 Backbone 输出层
    # 通常我们要找 output 附近的层，或者 blocks 的最后一层
    print("\n--- Searching for Output/Deep Layers ---")
    for key in state_dict.keys():
        # 筛选一些关键字，看看最深层的权重长什么样
        if 'head' in key or 'norm' in key:
            if 'weight' in key and len(state_dict[key].shape) > 0:
                pass  # 这里只是为了遍历

    # 打印最后几个层
    keys = list(state_dict.keys())
    for key in keys[-5:]:
        print(f"{key:<50} | {list(state_dict[key].shape)}")


if __name__ == "__main__":
    # inspect_config(config_path)
    inspect_weights(bin_path)