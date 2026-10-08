# af-record Toolkit v1.0.3

将 **af-record RAW**（`af_meta.json` + `af_rosbag/`）导出为完整 **af-record 标准格式**，包含 MP4、关节 JSON、MCAP、标注、LeRobot v2.1 / v3.0 与 RLDS。

## 版本说明

| 版本 | 文件 | 含义 |
|------|------|------|
| **工具包** `v1.0.3` | 根目录 `VERSION` | 本仓库软件发布号（脚本、依赖、导出能力） |
| **格式** `v1.0.2` | 根目录 `FORMAT_VERSION`、`af_meta.json` 的 `version` | af-record / af-raw 数据格式；相对格式 `v1.0.1` 新增必选目录 `af_lerobot_v3/` |

二者**独立演进**：修 bug / 优化导出可只升工具包；改目录结构或字段才升格式版本。

## 环境配置

要求 **Python 3.10+**。

```bash
cd af-record-toolkit-v1.0.3
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

依赖说明见文末「依赖」一节。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/verify_af_raw.py examples/sample-raw
python process_af_raw.py examples/sample-raw -o examples/sample-output
```

`-o` 省略则就地导出。RAW 规范见 [`docs/SPEC_RAW.md`](docs/SPEC_RAW.md)，集成见 [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md)，样例见 [`examples/README.md`](examples/README.md)。

## 标准输出

| 目录 | 说明 |
|------|------|
| `af_meta.json` / `af_rosbag/` | 元数据 + 原始 bag（自 RAW 复制） |
| `af_cameras/` / `af_joints/` / `af_mcap/` / `af_annotations/` | 派生产物 |
| `af_lerobot_v2/` / `af_lerobot_v3/` / `af_rlds/` | **必选** ML 数据集 |

## 工具包结构

```
af-record-toolkit-v1.0.3/
├── VERSION                 # 工具包版本
├── FORMAT_VERSION          # af-record / af-raw 格式版本
├── README.md
├── requirements.txt
├── process_af_raw.py       # RAW → 标准格式（主入口）
├── lerobot_v21_writer.py   # 内置 LeRobot v2.1 写入器
├── lerobot_v30_writer.py   # 内置 LeRobot v3.0 写入器
├── schemas/
│   └── annotations.schema.json
├── docs/
│   ├── SPEC_RAW.md         # af-raw 格式规范
│   └── INTEGRATION_GUIDE.md
├── scripts/
│   ├── verify_af_raw.py    # RAW 格式校验
│   └── build_samples.py    # 从 sample-raw 派生样例
└── examples/
    ├── README.md
    └── sample-raw/         # 唯一纳入 git 的基准 RAW
```

## 依赖与常见问题

`requirements.txt`：`rosbags`、`opencv-python-headless`、`imageio-ffmpeg`、`pyarrow`、`tensorflow`、`pymysql`、`minio`。无需 HuggingFace `lerobot`。

常见报错见 [`docs/SPEC_RAW.md`](docs/SPEC_RAW.md) 第 5 节。

## 版本

- **工具包 v1.0.3**：新增 `af_lerobot_v3` 导出、LeRobot 导出内存优化等
- **格式 v1.0.2**：标准输出相对格式 `v1.0.1` 增加必选 `af_lerobot_v3/`（LeRobotDataset v3.0）
