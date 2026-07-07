# 示例数据

除 **`sample-raw/`** 外，本目录下各 `sample-*` 样例均由脚本生成，**默认不纳入 git**（见仓库 `.gitignore`）。

## 生成

```bash
pip install -r requirements.txt
python scripts/build_samples.py
```

基准数据：`examples/sample-raw/`（3 相机 + 20 关节，159 帧）。脚本不会覆盖该目录。

## 样例列表

| 目录 | 相机 | 关节配置 | `verify_af_raw.py` |
|------|------|----------|---------------------|
| `sample-raw/` | 3 | 20 DoF（双臂 7+7 + 夹爪 + 头/身/升降） | PASS |
| `sample-1cam/` | 1 | 20 DoF（与 raw 相同） | PASS |
| `sample-4cam/` | 4 | 20 DoF | PASS |
| `sample-dual-arm-6joint-3cam/` | 3 | 双臂各 6 关节 + 夹爪（14 DoF） | PASS |
| `sample-single-arm-6joint-1cam/` | 1 | `arm_joint1`..`arm_joint6` | PASS |
| `sample-dual-arm-no-torso-3cam/` | 3 | 双臂 7+7 + 夹爪，无头/身/升降（16 DoF） | PASS |
| `sample-meta-mismatch/` | meta 4 / bag 3 | 20 DoF | FAIL |
| `sample-long-5min/` | 3 | 20 DoF，300 帧 @ 1fps（5 分钟时间轴） | PASS |
| `sample-invalid/` | — | 不合规 meta，无 bag | FAIL |

## 用法

```bash
python scripts/verify_af_raw.py examples/sample-single-arm-6joint-1cam
python process_af_raw.py examples/sample-dual-arm-6joint-3cam -o /tmp/out
```
