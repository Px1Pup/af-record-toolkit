# af-record RAW 数据格式要求

**格式版本：v1.0.1**（与 af-record 标准格式、af-record-toolkit 共用版本号。）

本文档描述 `process_af_raw.py` 所接受的**初始 raw 录制**格式：根目录下仅包含 `af_meta.json` 与 `af_rosbag/`。示例见 `examples/sample-raw/`。

---

## 1. 目录结构

输入目录（`input_dir`）必须同时满足：

| 路径 | 要求 |
|------|------|
| `af_meta.json` | 必须存在，合法 JSON |
| `af_rosbag/` | 必须存在 |
| `af_rosbag/metadata.yaml` | 必须存在 |
| `af_rosbag/*.db3` | 至少一个 SQLite3 bag 文件 |

**不应**在初始 raw 目录中包含以下衍生产物（由 `process_af_raw.py` 生成）：

- `af_cameras/`、`af_joints/`、`af_mcap/`、`af_annotations/`、`af_lerobot_v2/`、`af_rlds/`

---

## 2. `af_meta.json`

### 2.1 顶层字段

文件必须为合法 JSON，且包含以下**全部**顶层字段：

| 字段 | 类型 | 要求 |
|------|------|------|
| `robot_id` | string | 非空 |
| `robot_type` | string | 非空 |
| `author` | string | 非空 |
| `create_time` | string | 非空，建议 ISO 8601（如 `2026-05-20T10:36:41+08:00`） |
| `start_timestamp_ns` | number | > 0，与 bag 首帧时间戳一致 |
| `end_timestamp_ns` | number | > 0，≥ `start_timestamp_ns` |
| `duration_s` | number | ≥ 0 |
| `total_frames_count` | number | 正整数 |
| `data_validate` | object | `{ "validate": boolean, "reason": string }` |
| `fps_validate` | object | `{ "validate": boolean, "reason": string }` |
| `integrity_validate` | object | `{ "validate": boolean, "reason": string }` |
| `is_aligned` | boolean | 各流帧数是否一致；导出后由脚本更新 |
| `cameras` | array | **非空** |
| `end_effectors` | array | **非空** |
| `joint_names` | array | **非空**，顺序与关节话题一致 |
| `format` | string | 建议 `"af-record"` |
| `version` | string | 建议 `"v1.0.1"`（与 af-record / af-raw / toolkit 共版本号） |

### 2.2 `cameras[]`

每个元素必须包含：

| 字段 | 类型 | 要求 |
|------|------|------|
| `name` | string | 非空，用于匹配 rosbag 话题名 |
| `type` | string | 非空，相机型号描述 |
| `fps` | number | 正数，建议 30 |

**相机数量**：支持 1 路、3 路或更多；`name` 不固定，只要在 rosbag 话题名中能被唯一匹配即可。

**时间轴基准**：`cameras[0].name` 为 timeline 相机；其话题 `message_count` 必须等于 `total_frames_count`。

**LeRobot / RLDS 图像字段映射**（由 `process_af_raw.py` 自动推导）：

| af_meta 位置 | 导出字段名 |
|--------------|------------|
| `cameras[0]` | `image`（主视角） |
| `cameras[1..]` | 与 `cameras[i].name` 相同（若与保留字 `state`/`action`/`task`/`image` 冲突则加 `camera_` 前缀） |

示例：`cameras[0].name=main` → `image`；`cameras[1].name=wrist_left` → `wrist_left`。

### 2.3 `end_effectors[]`

每个元素必须包含非空 `name`、`type` 字符串。脚本导出时不直接读取该字段，但元数据完整性校验需要。

### 2.4 `joint_names[]`

- 每项为非空字符串
- 顺序即为导出 `af_joints/joint_states.json`、`af_lerobot_v2` 与 `af_rlds` 中 `state` / `action` 的维度顺序
- **必须与对应机器人本体 URDF 中的关节名称严格一致**（含大小写、下划线等），以便下游仿真、可视化与真机对齐
- bag 中关节话题的 `joint_names` 可与 meta 顺序不同，但**必须包含 meta 中的全部名称**

### 2.5 关节单位

标准 raw 中关节值**不再做单位换算**：

| 关节类型 | 单位 | 示例 |
|----------|------|------|
| 手臂关节（`left_arm_joint*`、`right_arm_joint*` 等） | **弧度（rad）** | `left_arm_joint1` |
| 夹爪关节（名称含 `gripper`） | **0~1 归一化** | `left_gripper_joint1` |
| 其他关节 | 按机型约定，一般为弧度 | `joint_head_yaw` |

---

## 3. `af_rosbag/`

### 3.1 存储格式

- ROS 2 rosbag2，**storage_identifier: sqlite3**
- **version: 9**
- 序列化格式：**cdr**
- 目录内包含 `metadata.yaml` 与一个或多个 `*.db3` 文件

### 3.2 必须包含的话题

| 用途 | 话题名规则 | 消息类型 | 说明 |
|------|------------|----------|------|
| 相机（每路一个） | 名称**包含**对应 `cameras[i].name` | `sensor_msgs/msg/CompressedImage` | 数量与 `cameras[]` 一致，每路唯一匹配 |
| 关节状态 | 名称**包含** `joint_states` | `control_interface/msg/JointState` | 建议 `/robot/joint_states` |

### 3.3 `CompressedImage` 要求

- `format` 字段为 JPEG 编码（如 `jpeg`、`jpg`）
- 载荷在 `data` 字段中
- 解码后为 BGR 彩色图（脚本导出 MP4 / LeRobot 时转为 RGB）

### 3.4 `JointState` 消息定义

```
std_msgs/Header header
string[] joint_names
float64[] joint_states
```

- `joint_names` 与 `joint_states` 长度必须相同
- `joint_states` 中手臂关节为弧度，夹爪为 0~1

### 3.5 帧数与时间对齐

**硬性校验**（`process_af_raw.py`）：

1. `cameras[0]` 对应话题的 `message_count` == `af_meta.json` 的 `total_frames_count`
2. 各相机话题消息数相同，且与关节话题消息数相同时，`is_aligned` 记为 `true`

**LeRobot / RLDS 导出**额外要求：

- `cameras[]` 中列出的**每路相机** + `joint_states` 须按**相同纳秒时间戳**一一对应
- 未对齐时导出失败并报错（`af_lerobot_v2` 与 `af_rlds` 为 af-record 标准格式的必选组成部分）

---

## 4. 导出（`process_af_raw.py`）

```bash
python process_af_raw.py <input_dir> -o <output_dir>
```

### 4.1 输出目录（af-record 标准格式）

```
<output_dir>/
├── af_meta.json
├── af_rosbag/
├── af_cameras/           # H.264 MP4
├── af_joints/            # joint_states.json
├── af_mcap/
├── af_annotations/
├── af_lerobot_v2/        # LeRobot v2.1 video 模式（必选）
└── af_rlds/              # RLDS TFRecord（必选）
```

`af_lerobot_v2` 与 `af_rlds` 导出失败时脚本报错退出。LeRobot 为 MP4 + parquet 路径引用（非 parquet 嵌像素）；`state`/`action` 与 meta `joint_names` 一致，`action` 为相邻帧差分。内置 `lerobot_v21_writer.py`，无需 HuggingFace lerobot。

---

## 5. 常见错误

| 现象 | 可能原因 |
|------|----------|
| `No CompressedImage topic whose name contains 'xxx'` | 话题名未包含 `cameras[].name` |
| `Multiple ... topics match camera` | `cameras[].name` 过短导致多话题匹配 |
| `total_frames_count does not match timeline camera` | meta 帧数与 `cameras[0]` 话题不一致 |
| `joint_names missing from bag joint topic` | meta 中关节名在 bag 消息里找不到 |
| `Incomplete aligned frame at timestamp ...` | 某路相机或关节在该时间戳缺失 |
| `af_rlds export requires tensorflow` | 未安装 `requirements.txt` 中的依赖 |
| `Native LeRobot v2.1 export requires pyarrow` | 未安装 pyarrow |

---

## 6. 参考

- 示例：`examples/sample-raw/`（`python scripts/build_samples.py` 生成其余样例）
- 集成：[`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md)
- 标注 schema：`schemas/annotations.schema.json`
