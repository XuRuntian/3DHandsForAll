# 统一多视角 MANO 输入输出 v1

编辑器只接受已完成多视角拟合、转换为统一约定的 MANO 参数。它不运行估计、三角化、同步、对齐、去畸变或重新优化。不接收单目弱透视结果、裸关节点、原生 PCA 或估计器私有目录。原单视角程序不变；旧 JSON 工具仍可离线使用，但不是新编辑器入口。

## 文件与职责

### 固定数据集目录的快捷启动

```bash
cd ~/3DHandsForAll
.venv/bin/python pose_annotator.py \
  ~/test_ego_anno/3D_Trajectory_Continuity_01
```

只传数据集根目录时，自动读取 `hand_annotations/inputs/manifest.json`，
优先加载 `hand_annotations/final/mano.npz`；尚无人工结果时加载
`hand_annotations/predictions/mano.npz`。输出默认是 `hand_annotations/final/`，
MANO 模型默认 `~/models/mano`，设备默认 CPU。缺少输入会明确报错，不自动运行估计。

原有 `--project`、`--input`、`--output`、`--mano-root` 均可覆盖默认路径。
显式 `--input` 优先于自动恢复人工结果；不要在想继续已有修正时传入旧预测。
指定自定义 `--output` 后，自动恢复只检查该输出目录。
下方更通用的 `predictions/<producer>/` 布局仍需用 `--input` 指定来源。

```text
project/
├── inputs/manifest.json          # 图片相对路径、帧对应及配套标定
├── predictions/<producer>/
│   ├── mano.npz                  # 独立适配器生成的统一预测
│   └── metadata.json
└── final/
    ├── mano.npz                  # 整段当前修正结果
    └── metadata.json
```

图像可以在任意已有目录；路径相对于 manifest.json。可以使用 `../`，不接受绝对路径。移动整个项目时保持相对关系；若需要删除估计包，先确保图像不再引用它。输入与输出的 MANO 格式相同。final 独立加载，不引用 predictions，也不保存网格、关节点或投影等可推导数组。

## 图像清单

UTF-8 JSON，`schema` 为 `multiview_images_v1`，字段为：

- `units`: `meter`。
- `extrinsic_convention`: `p_master = T_master_from_camera @ p_camera (homogeneous column vectors)`。
- `master_serial`: 主 RGB 相机序列号。
- `cameras`: 相机 ID 到标定的映射。当前 GUI 要求六台不同相机；每台包括 `serial`、`T_master_from_camera`（4×4 矩阵）及 `intrinsics`。
- `intrinsics`: `width,height,fx,fy,ppx,ppy,model,coeffs`。其中 `model` 是 `none`、`brown_conrady` 或 `inverse_brown_conrady`；`coeffs` 是五个畸变系数。内参/畸变必须对应实际读取的图片。去畸变图片使用处理后的内参及零畸变。
- `frames`: 按显示顺序排列的列表，每项含字符串 `frame_id` 和 `images`（相机 ID → 图片相对路径）。帧号唯一。某视角确实缺失时可省略其键或写 `null`；已列出的路径却不存在会报错，不猜测邻帧。

清单定义整段序列及其顺序，不靠连续数字或文件排序推测时间对应。数据准备方负责同步正确性。缺图视角显示提示，且不在占位画面叠加网格。

## mano.npz

仅含以下命名数组，不使用 pickle/object 数组。T 表示文件内帧数；固定手顺序为 `[right,left]`（被摄者解剖左右手）。

| 字段 | 形状 | 约定 |
|---|---|---|
| frame_ids | `[T]` | 唯一字符串帧号 |
| betas | `[2,10]` | 每只手整段共享的个体形状 |
| hand_pose | `[T,2,45]` | 15 个局部关节轴角，弧度，绝对姿态，flat_hand_mean=True |
| global_orient | `[T,2,3]` | 主相机坐标下的整体轴角，弧度 |
| transl | `[T,2,3]` | 主相机坐标下的平移，米；不是直接指定腕点 |
| valid | `[T,2]` | bool；true 表示完整可用的拟合参数，不表示人工验收 |

数值数组必须是有限浮点数；无效实例用有限占位值及 valid=false 表示，不进行渲染或编辑。不能缺少旋转/平移后伪装成有效手。手型可以是明确采用的零形状（平均手）；这不代表已估计了个体形状。

关节顺序为 MANO 原始顺序：食指、中指、小指、无名指、拇指，各指近端到远端。使用 MANO v1.2、原生模型原点（center_idx=None），先生成米制顶点，再加 transl。适配器负责 PCA 展开、hand mean、旋转表达、单位、坐标变换及原点差异；不靠编辑器猜测。

预测可以只覆盖部分帧并使用不同顺序；加载时根据 frame_ids 对齐到清单。未知帧报错，缺少帧填 valid=false。保存后的 final 覆盖完整图像清单，未修改的有效预测保持原值。逐帧不同的 betas 不能直接进入本规范，适配方须先确定每只手的一套共享形状。

## metadata.json

固定说明如下（序列号和 fingerprint 按实际输入填写）：

```json
{
  "schema": "mano_sequence_v1",
  "stage": "multiview_fitted",
  "model": "MANO_v1.2",
  "hand_order": ["right", "left"],
  "rotation": "axis_angle_radians",
  "hand_pose_convention": "absolute_flat_hand_mean",
  "translation_convention": "native_mano_origin",
  "units": "meter",
  "axes": "x_right_y_down_z_forward",
  "coordinate_system": "master_camera:338522300613",
  "image_manifest_sha256": "<manifest.json 文件字节的 SHA-256，64 位小写十六进制>"
}
```

指纹防止同样帧号/相机的不同片段被误配；只包含相对路径的清单整体搬家后无需改变。修改清单内容（含路径或格式）后须由数据准备方重新绑定参数文件，程序不会忽略不一致。metadata 在同一结果目录内保持不变。`stage` 是生产方对数据类型的声明，不是程序能独立证明的拟合质量。

## 保存与恢复

- 点击保存更新一份 `final/mano.npz`。临时文件写入并 fsync 后原子替换；失败不覆盖上一份参数文件。metadata 只初始化一次，不包含逐次变化的信息。
- 检测同一结果被另一窗口修改，拒绝静默覆盖。需要先关闭冲突窗口，重新载入最新结果，或另开输出目录。
- 切帧、切手或关闭窗口时保存已修改数据。没有手工修改时不强制创建 final；可点击保存初始化整段结果。
- 若输出目录已有 mano.npz，启动优先加载它。无需原始预测。恢复载入值指恢复本次加载时的该帧姿态及该手共享形状；它不是跨会话历史版本恢复。
- 修改 betas 影响当前手所有帧，另一只手不变；姿态/位姿只改当前帧。撤销/重做在当前帧内可撤销共享形状变化，切帧后重置撤销历史。
- 原始预测、原始图像和标定只读。不在本工具中删除 predictions。
