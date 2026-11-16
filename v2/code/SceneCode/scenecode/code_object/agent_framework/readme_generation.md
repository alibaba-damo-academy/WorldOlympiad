### Background target
设计一个 Agent Framework 来获取最高质量的 3D object 以及对应的 code pair 数据；
输入为 物体的 image，比如家具（凳子，椅子，桌子等）；输出为物体逐部件的blender代码，期待目录如下:
```
dataset_out/
  chair_xxx/
    image.png
    code/
      chair.py
      chair_parts/
        seat.py
        leg_01.py
        leg_02.py
        ...
    renders/
      full.png
      parts/
        seat.png
        leg_01.png
        leg_02.png
        ...
```

### Agent component
```
- Planner
  - 规划有哪些部件
  - Planner checker：部件数量对不对
  - 每个部件的大致 shape
- Part constructer
  - 构造结束后，马上检测是否构造的是否正确
  - 保存部件的代码
- Final check
  - 逐部件检测
  - 先检测物体的位置是否正确，check+modify 直到位置正确为止
  - 再处理形状，check+modify 直到形状正确为止
```

### Agent pipeline
配合 Blender-MCP，构建一个完整的 Agent pipeline；
agent_framework/
  agents/
    planner.py
        - 从输入图得到：类别（chair/table/stool）、关键部件列表、部件数量、每部件粗 shape 与相对关系（seat 在腿上、backrest 在 seat 后侧等）、层级（parent/child）。
        - 输出 ObjectPlan.json（结构化数据）
    planner_checker.py
        - 检查 关键部件列表 是否合理，部件是否重复、遗漏；若检查不通过，生成“修正指令”让 Planner 重写 plan
    part_constructor.py
        - 根据 ObjectPlan.json 和输入图片 构造部件，并保存部件的代码。
        - 保存 bpy 动作日志（Action Trace）：create_cube → scale → bevel → …（结构化 JSON）
    part_checker.py
        - render 出部件的 image，和输入 image 进行严格对比，修正 part 代码，直至验证通过
        - 逐部件检测；
        - 检测部件的位置是否正确，check+modify 直到位置正确为止
        - 检测部件的形状是否正确，check+modify 直到形状正确为止
  blender/
    暂时不用处理，借用 Blender-MCP 的代码，已经做到连接 Blender-MCP server，发送“执行 python / 获取场景信息 / 渲染图像”等指令
  eval（暂时不考虑，后期需要评估质量时再考虑）/
    image_metrics.py
    clip_score.py
    silhouette.py
  schemas/
    ObjectPlan.json：类别（chair/table/stool）、关键部件列表、部件数量、每部件粗 shape 与相对关系（seat 在腿上、backrest 在 seat 后侧等）、层级（parent/child）。
  prompts/
    planner.md
    constructor.md
    checker.md
  pipeline.py：Agent pipeline 的入口，负责调用各个 agent 组件，并保存结果
  config.yaml：配置文件，包括 参数、阈值、模型选择、路径、渲染设置、迭代预算 等
