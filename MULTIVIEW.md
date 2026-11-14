# 六视角 MANO 人工修正

## 职责与安装

本工具只读取统一图像清单和完整多视角 MANO 参数，负责渲染叠加、人工编辑、整段保存和继续编辑。同步/对齐由原始数据提供，去畸变和估计是独立链路。输入输出规范见 [MANO_INPUT.md](MANO_INPUT.md)。

使用 uv 管理 Python 3.10 和 Python 依赖。`pyproject.toml` 是依赖配置入口，`uv.lock` 锁定版本，`.python-version` 固定 Python 小版本。默认安装 CPU PyTorch，不需要 CUDA、PyTorch3D 或单视角预测权重。需要先安装 uv（至少 0.9.16）和 Git，并能够访问 PyPI、GitHub 和 PyTorch 下载源。

```bash
uv sync
uv run python pose_annotator.py --help
```

没有 Python 3.10 时 uv 会自动下载；无需先创建 conda 环境或手动激活 `.venv`。chumpy 的隔离构建依赖已在配置中声明，不需要手动安装 pip/setuptools。依赖更新使用 `uv lock`，严格复现锁文件使用 `uv sync --locked`。旧的 `requirements-multiview.txt` 仅保留给原有手动安装流程，uv 不读取它。

锁文件使用官方 PyPI 和 PyTorch CPU 源，配置方式参考 [uv 的 PyTorch 文档](https://docs.astral.sh/uv/guides/integration/pytorch/)。若终端设置了 `UV_DEFAULT_INDEX` 等镜像覆盖变量，普通 `uv sync` 可能重新解析并更新锁文件；严格复现时应清除这些覆盖或传入 `--default-index https://pypi.org/simple`。uv 0.9.16 可能提示 `extra-build-dependencies` 为实验选项，不影响安装。

已有 `.venv` 时，`uv sync` 会同步其中的包。若想保留旧环境（尤其是依赖 conda 或 CUDA 的环境），先将其改名，再同步：

```bash
mv .venv .venv.backup
uv sync
```

仅在 `.venv.backup` 不存在时执行以上备份命令；旧环境内脚本可能包含原路径，恢复使用时应先移回原名。不要在新版环境里运行旧版单视角 GPU 流程。

Tk 不是 PyPI 包。本次在 Linux 上验证了 uv 下载的 Python 3.10 自带 Tk；如使用系统 Python，须确保该解释器安装了匹配的 Tk 支持。`uv run python -m tkinter` 可检查窗口能否打开。GUI 仍需要可用的桌面显示，uv 不负责安装显示服务或系统图形库；Linux 若报告缺少 `libGL.so.1`，需通过系统包管理器安装对应库。

MANO 资产需自行取得授权，目录内包含 `MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl`，不随仓库分发，也不会由 `uv sync` 下载。图像、相机标定和预测参数同样需要另行准备。

## 通用启动

```bash
uv run python pose_annotator.py \
  --project /path/to/project/inputs/manifest.json \
  --input /path/to/project/predictions/model_v1/mano.npz \
  --mano-root /home/xrt/models/mano_v1_2/models \
  --output /path/to/project/final
```

`--output` 默认为清单目录的同级 `final/`。不指定 `--input` 时恢复已有 final；显式 `--input` 优先，继续编辑已有修正时不要传入旧预测。`--frame-id` 可指定清单中的帧号，默认打开清单第一帧。`--hand-side left/right` 控制初始编辑手。不能向 inputs 目录内写结果。

固定数据集目录可只传根目录启动，自动选择已有修正或初始预测，见 [快捷启动说明](MANO_INPUT.md#固定数据集目录的快捷启动)。

不带 --project/--multiview 仍使用旧单视角程序。原 `--multiview` 参数仅作为统一清单路径的别名；不再直接加载 HO-Cap 原生目录或旧的单帧 JSON。旧 JSON 文件不会被自动删除或覆盖。

## 独立 HO-Cap 转换示例

HO-Cap 只是适配器示例，编辑器没有对它的依赖，也不导入其模块。只支持带 conversion.json、拟合配置及完整验证网格的 sync_lab 导出包。适配器读取 PCA45 和 hand mean，验证全部有效帧网格后导出统一参数。无 GPU 估计或去畸变操作。

```bash
uv run python hocap_adapter.py /home/xrt/Downloads/hocap_hand_v1 \
  --mano-root /home/xrt/models/mano_v1_2/models \
  --export-project ./workspaces/hocap_review \
  --copy-images
```

目标必须是尚不存在的新目录。`--copy-images` 只复制已经去畸变的 RGB，不重新处理图像；得到的 inputs 独立于原包。省略该选项则用相对路径引用原包图像，原包图像必须保留。转换不会移动/修改原始数据。

本机已准备 `workspaces/hocap_review`，可直接运行：

```bash
uv run python pose_annotator.py \
  --project ./workspaces/hocap_review/inputs/manifest.json \
  --input ./workspaces/hocap_review/predictions/hocap_v1/mano.npz \
  --mano-root /home/xrt/models/mano_v1_2/models \
  --output ./workspaces/hocap_review/final
```

当前源数据是 3D_Trajectory_Continuity_01，259 帧。原包网格与转换网格最大差异约 0.00026 毫米；数值等价不证明原估计贴合正确。

## 编辑与保存

- 右侧「MANO 网格 / 21 点骨架」切换互斥显示模式；「叠加」可临时隐藏模型查看原图。透明度 0% 为完全显示、100% 为完全透明，默认 25%。六视图、专注视图和原图窗口共用显示模式与透明度，原图窗口可独立开关叠加。
- 21 点来自当前 MANO 的同一次前向计算，采用 OpenPose21 顺序：手腕 0，拇指 1–4、食指 5–8、中指 9–12、无名指 13–16、小指 17–20；指尖由 manopth 按左右手网格定义补齐。骨架显示 21 个点和 20 条连线，按手指着色，使用与网格相同的相机标定和米制三维坐标。它不是独立关键点预测，不支持直接拖点；仍通过左侧 MANO 参数编辑。显示设置不修改标注、不进入撤销记录，也不改变保存格式。
- 专注视图用于细调，六视图为 2×3 平铺。点击选相机、双击放大、Esc 返回六视图；滚轮缩放、拖动平移画面。完整显示保留整幅图，铺满模式会裁切边缘。
- 编辑区可拖动分隔条调整宽度；整体、手指、手型、精确四个标签，微调放在对应参数栏里。整体位置数值为毫米，旋转为轴角分量的度数。全局参数栏沿主相机坐标调整，不随选中相机改变。
- 方向键沿选中相机方向移动共享三维手，PgUp/PgDn 调深度；Shift 缩小到十分之一步长。在输入框/下拉框/滑块中操作时不触发移动快捷键。
- 六个视角显示同一只手；一次编辑一只手。没有有效预测的帧/手显示提示、禁用参数控件，不用默认手代替，但可从参考帧修复。缺图视角不叠加网格。
- 偶发误检可用“标记无手”：确认当前帧当前侧在整个场景中不存在，不是某个相机里不可见。错误姿态与漏检均可用“从参考帧修复”：选择其他同侧有效帧，查看六视角预览，确认后复制到当前帧，再进行必要微调。不需要手工标 21 点，不自动插值或重新识别。没有其他有效同侧帧时明确提示，不生成默认手。
- 标记无手、参考帧修复均支持撤销/重做和保存恢复；无手与原始漏检分别记录。修复不会改变共享手型或其他帧，复制结果仍需人工核对。
- 手型由当前手整段共享，姿态和平移只影响当前帧。按住微调按钮或连续拖动合并为一条撤销记录。Ctrl+Z 撤销，Ctrl+Y 重做，Ctrl+S 保存。
- 保存仅更新 final/mano.npz 与固定 metadata.json；不产生逐次 JSON 版本。最终参数不包含截图或推导网格。不指定其他输入重新打开时恢复 final，恢复载入值不依赖原始预测。

## 验证与边界

```bash
uv run python -m unittest discover -s tests -v
```

测试包括格式/单位/坐标校验、稀疏帧对齐、缺图与无效手、共享手型和撤销、原子写入失败及并发冲突、移走 predictions 后恢复、项目相对路径搬迁、六视图和真实 Tk 鼠标拖动回归，以及实际 HO-Cap 网格/投影和真实 MANO 编辑保存重载。缺本地资产/数据/显示时相关测试跳过；旧 Perspective 数据及 SDK 对照属于可选测试。

未实现自动估计、重新拟合、深度约束、无图三维编辑、跨会话撤销、HO-Cap 原生 PCA NPY 反向导出。旧单视角 GPU 优化流程未在本机验收。manopth/chumpy 的只读数组和弃用警告与本工具的滑块递归问题无关。
