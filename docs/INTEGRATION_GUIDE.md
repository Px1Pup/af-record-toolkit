# af-record 集成指南

客户自有格式 → **适配为 af-record RAW** → `process_af_raw.py` → af-record 标准格式。

## 边界

- 本工具**只读** RAW（`af_meta.json` + `af_rosbag/`），不解析客户私有格式
- **客户侧**负责适配脚本；字段与校验规则见 [`SPEC_RAW.md`](SPEC_RAW.md)
- RAW 合规后，一条命令导出完整标准格式（含 `af_lerobot_v2`、`af_rlds`）

## 适配要点

| 客户侧 | 写入 RAW |
|--------|----------|
| 各相机 JPEG 流 | `af_rosbag/` 中 `CompressedImage` 话题，话题名**唯一包含** `cameras[i].name` |
| 关节数组 | `joint_states` 话题；`af_meta.json` 的 `joint_names` 须全部被 bag 消息包含 |
| 帧数 / 时间戳 | `total_frames_count` = `cameras[0]` 话题帧数；ML 导出要求各流纳秒时间戳对齐 |
| 单位 | 手臂关节 **rad**，夹爪 **0~1**；非标准单位（如度×1000）在适配阶段换算 |

`cameras[]` 顺序与 `name` 决定 LeRobot/RLDS 图像字段名（`cameras[0]`→`image`，其余→`name`），详见 SPEC_RAW 2.2。

## 流程

```bash
pip install -r requirements.txt
python scripts/verify_af_raw.py <customer-raw>      # 先校验
python process_af_raw.py <customer-raw> -o <out>  # 再导出
```

示例与负例见 [`examples/README.md`](../examples/README.md)。

## 集成备忘

- bag 中**未列入** `cameras[]` 的图像话题会被忽略（不影响导出）
- `cameras[].name` 过短可能导致多话题匹配，应使用与话题名一致的具体名称
- LeRobot 输出为 **video 模式**（MP4 + parquet 路径），体积小属正常

完整格式定义、输出目录树与报错对照：[`SPEC_RAW.md`](SPEC_RAW.md)。
