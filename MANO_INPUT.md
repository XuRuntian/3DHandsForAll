# MANO 输入规范 v1：共享三维结果

**接入位置：多视角拟合之后、人工修正之前。** 输入是每帧每只手的一套共享 3D MANO，六台相机只负责观察和投影。不是各相机独立的 2D 检测、各相机各一套姿态、裸关节点或原始网络输出。

这个接口不绑定 HO-Cap：任何估计器经过轻量适配，输出符合下面约定的共享 MANO 都可接入。文件格式标识描述数据含义，不保证上游估计质量。当前未接入或运行估计器；已有针对 sync_lab 独立 HO-Cap 导出包的离线适配器，见 [MULTIVIEW.md](MULTIVIEW.md)。

## 1. 单帧单手：8 个必要字段

| 字段 | 内容 |
|---|---|
| `frame_id` | 数据集中同步帧的 ID，例如 `"0"`；不是各相机自己的采集帧号 |
| `hand_side` | `"right"` 或 `"left"`；被摄者的解剖左右手，不是画面左右 |
| `pose` | 15×3 或展平的 45 个数：15 个关节的局部轴角，单位弧度 |
| `betas` | 10 个 MANO 手型参数 |
| `global_orient` | 3 个数：全局轴角，单位弧度 |
| `transl` | 3 个数：主相机坐标中的平移，单位米 |
| `coordinate_system` | `"master_camera:<序列号>"`，本数据为 `"master_camera:338522300613"` |
| `units` | 必须写 `"meter"`，不自动猜测或转换单位 |

完整有效示例：[examples/mano_frame.json](examples/mano_frame.json)。示例中的零姿态仅展示格式，不是实际估计结果。

固定约定，不必逐次填写：标准 MANO v1.2 资产，10 维 Shape；pose 使用 MANO 原始顺序 **食指、中指、小指、无名指、拇指**，每指由近到远 3 个关节。pose 是绝对局部轴角（`flat_hand_mean=True`），不是 PCA 系数、欧拉角或相对 mean 的偏移。

坐标轴为主 RGB 相机 x 向右、y 向下、z 向前；模型用 `center_idx=None`，保持原生 MANO 原点，转为米后加 transl。**transl 不是直接指定腕关节位置**。Shape、模型版本和原点约定不同的估计器必须先做适配。

为兼容已有保存文件，仍接受原来的显式 `schema_version`、`mano_model_type`、`rotation_representation`、`pose_convention`、`translation_convention` 字段；省略时采用上述固定约定，明确填写不兼容值时拒绝导入。保存时自动补全这些约定，方便结果追溯。

## 2. 整段输入：同样的参数，外加一个帧列表

格式标识固定为 **`mano_multiview_v1`**，表示共享三维结果，而不是相机观测。

```text
schema: "mano_multiview_v1"
coordinate_system: "master_camera:338522300613"
units: "meter"
source: 可选，例如估计器名称和版本
frames:
  - frame_id: "0"
    hands:
      right: {pose, betas, global_orient, transl}
      left:  {pose, betas, global_orient, transl}
  - frame_id: "1"
    hands:
      right: {pose, betas, global_orient, transl}
```

上面是结构示意；可直接读取的 JSON 示例见 [examples/mano_multiview.json](examples/mano_multiview.json)。

- 左右手可同时提供，也可只提供一只。每个 frame_id 只出现一次，两只手放在同一个 hands 中。
- 没有可靠结果的手直接省略；两只手都缺失时省略该帧。不要填 `-1` 哨兵、NaN 或用零姿态冒充预测。格式无法判定数值合法的姿态是否可信，质量检查仍由适配器和人工承担。
- 可只导入一段或少量帧。每个 frame_id 必须存在于当前 RGB 序列；不靠数组下标猜对应关系。
- 切帧读取目标帧自己的预测。当前 UI 的上一帧/下一帧会跳过所选手没有预测的帧；若该帧没有另一只手，切换手时明确提示。
- 一帧一手的 Shape 要明确提供，即使整段相同。这样无需额外的被试表或跨文件依赖。

## 3. 谁负责转换

| 上游输出 | 适配器负责的转换 |
|---|---|
| HO-Cap 的 PCA 参数 | 按对应左右手 MANO basis 展开，并处理 hands_mean |
| 旋转矩阵或其他旋转表示 | 转成轴角，不能直接将欧拉角当轴角 |
| 毫米、其他世界/相机坐标 | 转为米和当前主相机坐标，同时正确处理全局旋转与模型原点 |
| 以腕点为中心或不同 hand mean | 按模型和 Shape 校正到统一原点、绝对局部姿态 |
| 只有 2D/3D 关键点、mesh 或弱透视参数 | 先完成 MANO 拟合或位姿恢复，不能直接读作本接口 |

估计器适配器应验证：转换前后产生的 3D 网格经过坐标变换后相同，六视角重投影一致。**仅通过 JSON 校验不等于几何转换正确。** 目前 `hocap_adapter.py` 支持带转换记录的 sync_lab HO-Cap 导出包，并验证全部有效帧网格；其他 HO-Cap 输出布局和 HaMeR 等仍需对应适配器。

后续嵌入流程：`运行估计器 → 适配器 → 本规范 JSON → 现有人工编辑器`。估计器可以作为后台进程运行，界面与保存格式不必绑定它的依赖或内部结构。

## 4. 导入与保存

```bash
# 只检查格式，无需 MANO 模型、GUI 或 GPU
.venv/bin/python mano_io.py examples/mano_multiview.json \
  --coordinate-system master_camera:338522300613

# 示例仅测试格式流程；正式使用时换成适配器导出的结果
.venv/bin/python pose_annotator.py \
  --multiview /home/xrt/Downloads/Perspective_Generalization_09/rgbd \
  --mano-root /home/xrt/models/mano_v1_2/models \
  --input examples/mano_multiview.json \
  --output ./annotations_multiview
```

窗口的 加载结果 也接受整段文件、单帧记录或已保存的修正文件。整段导入时优先选择当前帧和手；不存在时选择数据顺序中的首个可用记录。导入新整段文件前，窗口会保存当前已修改的内容。

保存仍为独立版本文件，包含完整 `original`、`corrected`、输入来源；不会覆盖估计器结果。会话内切帧保留已编辑状态；重启后先导入预测序列，再用 加载结果 载入需要继续编辑的修正版本。当前没有自动批量合并历史修正或批量导出整段的功能。
