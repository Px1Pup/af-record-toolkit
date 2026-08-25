# af-record Toolkit v1.0.2

将 **af-record RAW**（`af_meta.json` + `af_rosbag/`）导出为完整 **af-record 标准格式**，包含 MP4、关节 JSON、MCAP、标注、LeRobot v2.1 与 RLDS。

**af-record 格式、af-raw 格式与本工具包共用版本号**（见根目录 `VERSION` 与 `af_meta.json` 的 `version` 字段）。

## 环境配置

要求 **Python 3.10+**。

```bash
cd af-record-toolkit-v1.0.2
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
| `af_lerobot_v2/` / `af_rlds/` | **必选** ML 数据集 |

## 工具包结构

```
af-record-toolkit-v1.0.2/
├── VERSION                 # 与 af-record / af-raw 共版本号
├── README.md
├── requirements.txt
├── process_af_raw.py       # RAW → 标准格式（主入口）
├── lerobot_v21_writer.py   # 内置 LeRobot v2.1 写入器
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

`requirements.txt`：`rosbags`、`opencv-python-headless`、`imageio-ffmpeg`、`pyarrow`、`tensorflow`。无需 HuggingFace `lerobot`。

常见报错见 [`docs/SPEC_RAW.md`](docs/SPEC_RAW.md) 第 5 节。

## 版本

**v1.0.2** — af-record / af-raw / toolkit 共版本号（`VERSION`、`af_meta.json` 的 `version` 字段）。
