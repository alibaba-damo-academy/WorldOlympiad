# T4: 6000D 干净环境安装验证（2026-11-16）

**结论：限定通过。** 在 6000D 的全新本地目录中，从候选源码新建了五套 Python 环境，构建了 gsplat/cuRobo CUDA 扩展，完成仓库运行时检查、SAPIEN 相机读回及 Blender 一帧渲染。安装受到共享 GPU 争用中断，随后使用同一候选中固定的 bootstrap 脚本续装；**没有得到单条 `runtime --online-install` 无中断退出 0 的证据**。测试对象为 LingBot-VA 的已记录部署 profile，T5 的新环境完整评测依用户要求暂缓。

## 测试对象与复用边界

- 源码提交 `7518d72d995db702f4f0e336b73a539a094e400e`，Git tree `65e6b261721d8f6896c9c5327f5cc8e504a0edc1`；归档 SHA256 `d3f05d3ededa8a57610818ff043237254f35c67546d5bf0ce53faeec204b4f54`。七个源码包均通过清单验证。
- 五套环境 `autoresearch`、`scenecode`、`worldeval`、`robotwin`、`lingbot` 均在独立目录新建；未链接旧环境或旧 runtime overlay。只复用已下载的评测权重、LingBot checkpoint、RoboTwin 数据资产、Blender 归档、Python/uv/CUDA 工具链和下载缓存；未下载或替换模型权重。
- 主机内核驱动 580.173.02、CUDA toolkit 12.8，测试 GPU 的实际架构为 `sm_120`。CUDA toolkit 的头文件和库位于 `targets/x86_64-linux/{include,lib}`；构建时在本轮 shell 中设置 `CPATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH` 和兼容的主机编译器。未修改系统配置或共享环境。
- 6000D 缺少系统 NVIDIA EGL/Vulkan 用户态图形组件。Blender 走主机 Mesa EGL；SAPIEN 使用与内核驱动匹配、用户可读的 NVIDIA Vulkan bundle，并通过进程内 `VK_DRIVER_FILES`、`VK_ICD_FILENAMES`、`LD_LIBRARY_PATH` 配置。该 bundle 是主机前置条件，不随仓库分发。

## 执行证据

原始日志和退出码保存在隔离安装目录的 `outputs/install_validation_20261115/`，不进入 Git。

| 检查 | 结果 |
|---|---|
| `sources-verify`、RoboTwin 资产、评测权重及 LingBot checkpoint | 全部通过；原权重未改写 |
| 首次 `./setup.sh runtime --online-install` | 01:16 获得空卡并启动；autoresearch、SceneCode、WorldEval 安装成功，gsplat CUDA 扩展及评测导入通过；01:23 下一次 GPU 门禁遇其他作业，退出 1，保留已完成环境 |
| RoboTwin 续装 | 固定 bootstrap 脚本完成依赖和 cuRobo 构建；首次 SAPIEN smoke 因未传入 NVIDIA Vulkan bundle 而失败。使用既有 host bundle 的进程内配置重跑同一 smoke 退出 0，Torch 2.9.0+cu128、CUDA 计算、五个 cuRobo 扩展和 SAPIEN 渲染器通过 |
| LingBot-VA 续装 | 全新环境安装及 Torch CUDA、模型导入 smoke 退出 0 |
| `./setup.sh runtime --check-only` 与 `./setup.sh runtime` | 两条命令均退出 0；五套环境的解释器版本、核心导入和三个 runtime overlay 通过 |
| SAPIEN 相机与 Blender | 新 RoboTwin 环境的 32×32 RGBA 相机读回退出 0；Blender 4.4.3 在 Mesa 路径完成 32×32 PNG 渲染，退出 0。Blender 日志包含 `EGL_BAD_MATCH` 警告，但已写出可读图像 |
| 磁盘与进程 | 安装产物位于服务器本地 `/home` 文件系统；未向其他用户进程发送信号；测试 GPU smoke 退出后释放本轮进程 |

## 依赖审计与结论边界

本轮保存了五套环境的完整 `uv pip freeze` 清单。`autoresearch`、`robotwin`、`lingbot` 的 `uv pip check` 通过。`scenecode` 和 `worldeval` 的通用 `pip check` 分别报告 11 和 4 项上游包元数据缺项，不能当作全依赖兼容通过：

- SceneCode 的当前 `pyproject.toml` 比冻结的 `uv.lock` 声明更多包；部署脚本按 `uv.lock` 的所选 extras 安装，并单独安装 `bpy`。缺项包含开发工具、未使用的无头 VTK 分支所需 `pyvirtualdisplay`、`mathutils` 发行包及 `bpy` 元数据依赖；本轮实际 SceneCode 导入和 Blender 渲染通过。此锁与上游元数据差异仍须在宣称完整包锁前解决。
- DA3 上游元数据包含 `xformers` 等应用依赖。部署脚本有意以 `--no-deps` 安装源码，避免其不受约束的依赖替换固定的 Torch/cu128 评分栈；WorldEval 核心导入和 CUDA 扩展 smoke 通过。未测试 DA3 上游应用的所有可选功能。
- `runtime.lock.json` 锁定关键版本、源码提交和 CUDA wheel profile，但并未固定所有直接与传递依赖及 wheel 哈希。本轮 `freeze` 是安装结果记录，**不是跨机器完整锁文件**。

因此本次证据支持“6000D 上同机干净环境按固定源码和工具链可安装并完成运行时、CUDA、Vulkan/Mesa 检查”。它不证明无人争用时的一条命令安装、5090 上当前提交的回归、所有模型 adapter、完整新环境评测或任意新服务器的直接部署。公开发布仍须完成仓库许可证和第三方源码分发权审查。
