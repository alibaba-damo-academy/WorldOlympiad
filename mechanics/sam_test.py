"""
Docstring for mechanics.sam_test
config:
    model的位置
    cuda编号
    视频的位置
    检测目标的文本提示词
    conf, iou等参数
"""

from ultralytics.models.sam import SAM3VideoSemanticPredictor
import numpy as np

# Initialize semantic video predictor
overrides = dict(conf=0.5, iou=0.7, task="segment", mode="predict", imgsz=640, model="./weights/sam3/sam3.pt", half=True, save=True, device='cuda:1', format='mp4')
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# Track concepts using text prompts
results = predictor(source="data/videos/buoyancy_self_forcing_2.mp4", text=["one whole lemon"], stream=True)


for frame_idx, r in enumerate(results):
    print(f"--- Frame {frame_idx} ---")
    
    # 检查是否有掩码被检测到
    if r.masks is not None:
        # 获取所有掩码的张量数据 (N, H, W)
        masks = r.masks.data.cpu().numpy()
        
        # 如果有类别标签，可以一起遍历
        # r.boxes.cls 获取类别 ID, r.names 获取名称映射
        classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else []

        for i, mask in enumerate(masks):
            # 找到掩码中所有非零（物体部分）的坐标
            coords = np.argwhere(mask > 0)
            
            if len(coords) > 0:
                # np.argwhere 返回的是 [row, col] 即 [y, x]
                # 计算平均值得到质心
                center_y, center_x = coords.mean(axis=0)
                
                label = r.names[int(classes[i])] if i < len(classes) else f"Object {i}"
                print(f"Detected: {label} | Centroid: x={center_x:.2f}, y={center_y:.2f}")
            else:
                print(f"Object {i}: Mask is empty")
    else:
        print("No objects detected in this frame.")
