# 六视角 MANO 人工修正

本扩展继承 `pose_annotator.py` 的 Tk 窗口类，直接复用 `init_sliders`、手指/旋转/平移回调和 freeze/restore 逻辑。单视角入口保持原行为。六视角不加载预测网络，不运行优化，不读取深度约束。原 `mano_wrapper.py` 的 CUDA、60° FOV、右手模型、10 cm 骨长归一化不能直接用于米制标定，因此新增轻量适配器，仍使用项目原有的 `manopth.ManoLayer`，不修改旧优化器。

## 本机检查结果

- `rgbd/frames.jsonl` 提供 253 组配对帧，ID 0–252；按 JSON 中的 `color_path` 加载，不能用原采集 frame_number 直接配对。
- 六路 `cam00`–`cam05/color_000000.jpg` 等，RGB JPEG 1280×720；所有 1518 张 RGB 已成功解码。
- 编号/序列号：cam00=338522302174，cam01=246322300746，cam02=342622301386，**cam03=338522300613（主相机）**，cam04=341222301246，cam05=342622302051。
- 外参来自 `rgbd/calibration/extrinsics.json`，明确声明 `p_master = T_master_from_camera @ p_camera`，齐次列向量、米；投影使用其逆。相机坐标 x 向右、y 向下、z 向前。不使用 depth_to_color 外参，因为输入已是 RGB。
- K/D 来自每个 `camXX/calibration.json` 的 `original_calibration.intrinsics.color`，对应当前 RGB。六路均为 inverse Brown-Conrady。投影公式与本机 `multical/multical/camera_realsense.py`、`realsense-sync-lab/sync_lab/hocap_sequence.py` 和 [RealSense SDK](https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp) 一致；不是把 D 直接交给普通 OpenCV 模型。
- **标定文件存在差异**：rig 的 `calibration/calibration.json` 内 K 与逐机录制 K 最大元素差为 0.35–1.06 像素；本工具选择逐机录制 K，并原样使用 rig 外参，不擅自重新标定。实际跨视角贴合精度仍须人工核验。
- 配对 span 为约 1.439–1.490 ms。数据 manifest 明确说这不是物理曝光同步精度保证；rig report 也提示平面单标记外参歧义，0.723 px 标定残差不能证明外参绝对正确。
- 用户已提供 `/home/xrt/models/mano_v1_2/models/MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl`；真实左右手均已成功加载，778 个顶点、1538 个三角面。缺失所选手模型时仍明确报错。

## 安装与启动

使用带 Tk 的 Python 3.10（规避旧 manopth/chumpy 与新版 NumPy/Python 的兼容问题）。本次已在仓库 `.venv` 准备依赖；模型已使用用户下载的 MANO v1.2。

```bash
cd /home/xrt/3DHandsForAll
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python pip setuptools wheel
uv pip install --python .venv/bin/python 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python --no-build-isolation -r requirements-multiview.txt

# 此目录下应直接包含 MANO_LEFT.pkl 和 MANO_RIGHT.pkl
.venv/bin/python pose_annotator.py \
  --multiview /home/xrt/Downloads/Perspective_Generalization_09/rgbd \
  --mano-root /home/xrt/models/mano_v1_2/models \
  --hand-side right --frame-id 0 \
  --output /home/xrt/3DHandsForAll/annotations_multiview
```

仅有一个手模型也可以启动该手；切换另一手会提示缺失资产。`--device cpu` 为默认，支持 `--device cuda`（需相应 PyTorch）。无 `--multiview` 时走原单视角代码，仍需要原项目依赖和资产。输出目录必须在原数据目录之外。

不需要 MANO 的数据检查与测试：

```bash
.venv/bin/python multiview_data.py /home/xrt/Downloads/Perspective_Generalization_09/rgbd --decode-all
.venv/bin/python -m unittest discover -s tests -v
# 安装可选 pyrealsense2 后，SDK 一致性测试也会执行；否则该项显示 skip。
```

## 操作（新版中文工作区）

- 默认“专注视图”：右侧显示选中相机大图，底部为六路缩略图；单击缩略图切换相机。下拉菜单可切换为 2×3 六视图；双击画面进入专注视图，Esc 返回六视图。
- 图像随窗口缩放，保持宽高比。“完整显示”不裁切，容器比例不同时会有留边；“铺满（裁切）”填满容器但可能裁掉边缘，界面会持续提示。图像和网格采用相同显示变换，不改变相机标定或三维参数。
- 所有画面支持滚轮缩放、拖动平移，“复位缩放”恢复选中画面。原图窗口也自适应大小，并提供“适应窗口／原始尺寸”。
- 左侧编辑区默认宽 480 像素，可拖动中间分隔条调整宽度。分为“整体、手指、手型、精确”四组；滑块轨道和拖动柄已加大。手指组使用五个按钮直接切换；继续复用原项目的关节滑块及快照恢复功能。整体滑块旁显示实际轴角分量（度）与位置（毫米），右侧数值框可直接输入、回车应用；滑块映射保持原样。
- 微调直接放在对应参数栏内，不再设置独立页：整体旋转与位置各栏提供 0.1 / 0.5 / 1 / 5 / 10 度或毫米步长和加减按钮，修改与相邻数值框一致的主相机坐标分量（旋转为轴角分量）。手指各滑块每次加减 1 格，手型参数每次加减 0.01；按住按钮可连续调整，一次按住作为一条撤销记录。六视图仍共享同一个三维手。
- “精确”页可读取或修改全部 45 个 pose 分量、整体旋转/位置、10 个 Shape 系数，编号从 0 开始；精确输入与 JSON 的位置单位为米，旋转单位为弧度。
- 顶部支持上一帧、下一帧、输入帧号跳转、切换左右手。导入预测序列后，上一帧/下一帧跳过所选手没有预测的帧；直接跳转到缺失预测的帧会提示。
- 撤销/重做仅针对当前帧当前手，保留最近 100 步；一次连续拖动滑块合并为一步。切帧、切手或加载结果时重新开始撤销记录。快捷键：Ctrl+Z 撤销，Ctrl+Y 重做，Ctrl+S 保存；方向键按相机方向移动，PgUp/PgDn 靠近/远离，Shift 将移动步长缩小为十分之一。焦点在数值输入框、下拉框或滑块时，不触发这些移动快捷键。
- “保存修正”写入新版本；切帧、切手、加载另一文件、关闭窗口前自动保存已修改内容。“恢复初值”恢复原始预测/默认初值，本身也可以撤销。输入预测文件不被覆盖。
- 默认平手位置仅依据标定光轴交汇点初始化，不是对真实图中手的估计。网格仍是线框显示，不进行深度遮挡剔除或多视角重新优化。

## 统一 JSON 接口（v1）

可直接通过 `--input prediction.json` 或 加载结果 导入以下独立对象，也可导入本工具保存的 annotation 包。严格检查当前 frame_id 与主相机 serial；不推断陌生模型的 PCA、单位或坐标系。

| 字段 | 约定 |
|---|---|
| schema_version | 1 |
| frame_id | 字符串；与 frames.jsonl 中 ID 的字符串形式完全一致 |
| hand_side | `left` / `right`，使用对应真实 MANO 资产，禁止用翻转图像替代 |
| pose | 长度 45，15×3 展平的局部轴角，弧度；MANO 顺序 index、middle、pinky、ring、thumb，每指 3 关节 |
| global_orient | 长度 3，全局轴角、弧度；不含在 pose 内 |
| betas | 长度 10，MANO Shape 系数 |
| transl | 长度 3，米；MANO 原生模型原点的平移（不是强行居中的腕关节位置） |
| coordinate_system | `master_camera:338522300613`，所有姿态/平移已转换至主 RGB 相机坐标 |
| units | `meter` |
| mano_model_type | `MANO` |
| rotation_representation | `axis_angle_radians` |
| pose_convention | `absolute_flat_hand_mean`；absolute 局部轴角，不是 PCA，也不是相对 MANO hands_mean 的偏移 |
| translation_convention | `native_mano_origin` |

可用 `mano_io.default_parameters('0', 'right', 'master_camera:338522300613')` 生成完整对象。manopth `center_idx=None, flat_hand_mean=True, use_pca=False`；输出顶点由毫米转米，再加 transl，不做旧版骨长缩放。对于 mean-offset 模型，适配时先加该侧 hands_mean；对于矩阵旋转/PCA/弱透视相机输出，需要上游显式转换。HaMeR/EasyMocap 未接入、未运行。

每次保存为 `<frame_id>_<hand_side>.v0001.json` 等，独占创建，不覆盖历史文件：

```json
{
  "schema": "3dhands.multiview.annotation.v1",
  "source": "/absolute/path/to/input.json",
  "original": {"...": "完整原始 MANO 对象"},
  "corrected": {"...": "完整人工修正对象"}
}
```

原始预测文件不被修改；重新载入 annotation 包时继续保留最早 original。左右手独立保存，每次编辑一个手。

## 验证范围

已验证真实数据目录、全部 RGB 解码、六台相机外参逆变换、米制平移、SDK 畸变一致性、相机后方裁除、JSON 校验与精确往返、原始预测不覆盖及版本化保存。

新增 `tests/test_multiview_mano_assets.py` 使用用户提供的真实左右手模型，验证实际网格尺寸/拓扑、默认顶点全部位于六路画面内、pose/global_orient/betas 编辑确实改变几何、米制平移、保存重载后顶点逐元素一致，以及真实 Tk 窗口中的编辑、左右手切换、恢复和六视图刷新。12 项测试在本仓库环境执行：11 项通过，1 项可选 SDK 测试因该环境未安装 pyrealsense2 跳过；该 SDK 项已在已有 RealSense 环境通过。可通过 `MANO_TEST_ROOT` 指定测试模型目录，未提供资产的机器会跳过真实模型测试。

CPU 六视图加一个 1.5× 原图窗口，10 次更新平均约 107 ms/次（约 9 次/秒，仅本机测量，不保证所有设备性能）。旧 manopth/chumpy 有弃用及只读数组提示，但此次加载和编辑测试未失败，未修改模型资产或第三方安装包。

**仍未验收：对图中真实手进行逐帧人工贴合后的跨视角标定精度，及旧单视角 GPU/优化流程。** 默认平手只用于流程验证，不是估计结果。

## 简化的通用输入（更新）

主要接入口已明确为**多视角拟合后的共享 3D MANO 结果**。单帧输入只需 8 个必要字段，支持整段 `mano_multiview_v1` JSON；旧的完整参数对象和修正版本保持兼容。详细约定、示例和适配器责任以 [MANO_INPUT.md](MANO_INPUT.md) 为准。上文列出的固定约定字段可由程序自动补全，不再要求估计器逐次填写。

六视角窗口已中文化：按钮、手指滑块、手型参数、参数选项、状态、查看原图窗口和常见错误提示使用中文。参数文件字段及原单视角界面保持原有约定。已打开的旧窗口需先保存并关闭，再用原命令启动。

### 本次界面改版验证

新增 `multiview_layout.py`（布局）和 `multiview_viewport.py`（等比例视口），保持 MANO 参数和原滑块回调兼容。已实际检查大图、手指编辑页及六宫格截图；新增视口缩放/裁切几何、窗口缩放、控件分组可见性、选中相机方向的毫米位移、撤销重做及拖动合并测试。当前 20 项测试中 19 项通过，1 项可选 RealSense SDK 测试跳过。真实 MANO 六视图加原图窗口渲染约 142 ms/次（本机 CPU 测量，不含估计器）。

### 滑块回调递归修复

修复手指滑块拖动时，程序同步关联滑块再次触发 Tk Scale command，导致 `RecursionError` 的问题。多视角模式在滑块回调入口阻止同步期间的编辑，并避免同步函数嵌套刷新 idle 事件；保留同步前的暂停状态。单视角回调不变。

回归测试通过真实 Tk 鼠标按下、拖动和释放事件，覆盖左右手五根手指的整体弯曲和第一关节滑块，以及撤销、反复同步后导入参数精度不变。完整测试共 21 项，20 项通过，1 项可选 RealSense SDK 测试跳过。启动时 manopth 的只读 NumPy 数组警告仍可能出现，和本次回调递归是不同问题。

## 导入本机 HO-Cap 导出包

已支持 `prepare_hocap_sequence.py` 生成的独立、去畸变、主相机坐标数据包（包含 `conversion.json` 和完成后的 `05_coordinates/mano_world.npz`）。这不是所有 HO-Cap 原生目录的通用加载器。此包来自 `3D_Trajectory_Continuity_01`，不要搭配原来的 Perspective 图像。

```bash
.venv/bin/python pose_annotator.py \
  --multiview /home/xrt/Downloads/hocap_hand_v1 \
  --mano-root /home/xrt/models/mano_v1_2/models \
  --hand-side right \
  --output ./annotations_hocap
```

未指定 `--input` 时自动导入包内最终拟合结果；CPU 即可。使用本包 rectified RGB 与零畸变内参，通过转换记录核对外参方向、主相机和米制单位。使用原始帧索引与导出帧的显式映射，不读取旧机器的绝对路径。

适配器按拟合配置将 PCA45 展开并加对应手的 MANO mean，转换为统一绝对轴角；保留原生 MANO 原点和米制平移。每次导入验证全部有效帧的重建网格与包内 `vertices_world`，最大误差超过 0.01 毫米则拒绝导入，避免不兼容模型/约定静默错位。无效手不填默认姿态。

当前包验证：259 帧、518 套有效手参数；最大网格差异约 0.00026 毫米。所有六路投影与输出 `joints_2d` 数值一致性测试通过。此检查确认导入一致性，不证明估计已经贴合真实手。

操作：选帧 → 切换左/右手 → 选择相机并放大 → 在整体/手指栏修正 → 保存。六视角仍共用一套参数。修正另存到 `annotations_hocap`，包括原始预测和修正值，原包只读。继续某一帧修正时，以相同命令启动、跳到对应帧，再用“加载结果”选择保存的 JSON。当前尚不自动合并整段历史修正。

也可只导出统一 JSON（输出文件必须不存在，且位于原包之外）：

```bash
.venv/bin/python hocap_adapter.py /home/xrt/Downloads/hocap_hand_v1 \
  --mano-root /home/xrt/models/mano_v1_2/models \
  --output ./annotations_hocap/imported_predictions.json
```

新增依赖 `PyYAML>=6,<7`，已列入 requirements-multiview.txt；没有接入估计器、重新优化或深度约束。测试 `tests/test_hocap_import.py` 覆盖实际包网格/投影、错误配置/模型拒绝、真实 Tk 编辑及保存重载；无本地包或 MANO 资产时跳过。可用 `HOCAP_TEST_ROOT` 指定测试数据包。
