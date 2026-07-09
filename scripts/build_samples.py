#!/usr/bin/env python3
"""Generate af-record RAW example datasets from examples/sample-raw/."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import StoragePlugin, Writer

ROOT = Path(__file__).resolve().parent.parent
BASE_SAMPLE = ROOT / "examples" / "sample-raw"
EXAMPLES_DIR = ROOT / "examples"
AF_RECORD_VERSION = "v1.0.1"

MSG_COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"

CAMERA_TOPICS = {
    "head_color": "/camera/head_color/image/compressed",
    "hand_left_color": "/camera/hand_left_color/image/compressed",
    "hand_right_color": "/camera/hand_right_color/image/compressed",
}
EXTRA_CAMERA = ("chest_color", "/camera/chest_color/image/compressed")
JOINT_TOPIC = "/robot/joint_states"

THREE_CAM_NAMES = ("head_color", "hand_left_color", "hand_right_color")

# Base bag joint order (20 DoF).
BASE_JOINT_NAMES = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
    "left_gripper_joint1",
    "right_gripper_joint1",
    "joint_head_yaw",
    "joint_head_pitch",
    "joint_body_pitch",
    "joint_lift_body",
]

TORSO_JOINTS = (
    "joint_head_yaw",
    "joint_head_pitch",
    "joint_body_pitch",
    "joint_lift_body",
)

JOINT_NOISE_STD_RAD = 0.008


@dataclass(frozen=True)
class RobotProfile:
    sample_name: str
    record_id: str
    robot_type: str
    joint_names: tuple[str, ...]
    cameras: tuple[str, ...]
    base_joint_map: dict[str, str]


# Joint names match each robot URDF; hardcoded here so generation only needs sample-raw/.
ROBOT_PROFILES: tuple[RobotProfile, ...] = (
    RobotProfile(
        sample_name="sample-single-arm-6joint-1cam",
        record_id="SAMPLE-LHC-LA-L2-SL",
        robot_type="lhc_la_l2_1050_description_sl",
        joint_names=tuple(f"r_joint{i}" for i in range(1, 7)),
        cameras=("head_color",),
        base_joint_map={f"r_joint{i}": f"left_arm_joint{i}" for i in range(1, 7)},
    ),
    RobotProfile(
        sample_name="sample-dual-arm-6joint-3cam",
        record_id="SAMPLE-LHC-DA2",
        robot_type="lhc_da2_description_2603",
        joint_names=tuple(
            [*(f"r_joint{i}" for i in range(1, 7)), *(f"l_joint{i}" for i in range(1, 7))]
        ),
        cameras=THREE_CAM_NAMES,
        base_joint_map={
            **{f"l_joint{i}": f"left_arm_joint{i}" for i in range(1, 7)},
            **{f"r_joint{i}": f"right_arm_joint{i}" for i in range(1, 7)},
        },
    ),
    RobotProfile(
        sample_name="sample-dual-arm-lift-pitch-3cam",
        record_id="SAMPLE-LHC-IDA-L2-PITCH",
        robot_type="lhc_ida_l2_pitch_800_description",
        joint_names=tuple(
            [*(f"l_joint{i}" for i in range(1, 7)), *(f"r_joint{i}" for i in range(1, 7))]
        ),
        cameras=THREE_CAM_NAMES,
        base_joint_map={
            **{f"l_joint{i}": f"left_arm_joint{i}" for i in range(1, 7)},
            **{f"r_joint{i}": f"right_arm_joint{i}" for i in range(1, 7)},
        },
    ),
    RobotProfile(
        sample_name="sample-dual-arm-folding-3cam",
        record_id="SAMPLE-LHC-RW2-ZP-6G",
        robot_type="LHC-RW2-ZP-6G_description",
        joint_names=tuple(
            [*(f"l_joint{i}" for i in range(1, 7)), *(f"r_joint{i}" for i in range(1, 7))]
        ),
        cameras=THREE_CAM_NAMES,
        base_joint_map={
            **{f"l_joint{i}": f"left_arm_joint{i}" for i in range(1, 7)},
            **{f"r_joint{i}": f"right_arm_joint{i}" for i in range(1, 7)},
        },
    ),
)


def load_meta_template() -> dict:
    return json.loads((BASE_SAMPLE / "af_meta.json").read_text(encoding="utf-8"))


def read_base_messages() -> tuple[dict[str, list[tuple[int, bytes]]], dict[str, str], object]:
    bag_dir = BASE_SAMPLE / "af_rosbag"
    messages: dict[str, list[tuple[int, bytes]]] = {}
    msgtypes: dict[str, str] = {}
    with AnyReader([bag_dir]) as reader:
        typestore = reader.typestore
        for conn in reader.connections:
            msgtypes[conn.topic] = conn.msgtype
            messages[conn.topic] = [
                (timestamp, bytes(raw))
                for _, timestamp, raw in reader.messages(connections=[conn])
            ]
    return messages, msgtypes, typestore


def write_rosbag(
    out_dir: Path,
    topic_messages: dict[str, list[tuple[int, bytes]]],
    msgtypes: dict[str, str],
    typestore: object,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    with Writer(out_dir, version=9, storage_plugin=StoragePlugin.SQLITE3) as writer:
        connections = {
            topic: writer.add_connection(topic, msgtypes[topic], typestore=typestore)
            for topic in topic_messages
        }
        for topic, rows in topic_messages.items():
            for timestamp, raw in rows:
                writer.write(connections[topic], timestamp, raw)

    db3_files = sorted(out_dir.glob("*.db3"))
    if len(db3_files) != 1:
        raise RuntimeError(f"expected one db3 in {out_dir}, found {len(db3_files)}")
    target = out_dir / f"{out_dir.name}_0.db3"
    if db3_files[0].name != target.name:
        db3_files[0].rename(target)
        meta_path = out_dir / "metadata.yaml"
        meta_path.write_text(
            meta_path.read_text(encoding="utf-8").replace(db3_files[0].name, target.name),
            encoding="utf-8",
        )


def build_timeline(frame_count: int, fps: float, start_ns: int) -> list[int]:
    step_ns = int(round(1_000_000_000 / fps))
    return [start_ns + index * step_ns for index in range(frame_count)]


def resample_messages(
    source_rows: list[tuple[int, bytes]],
    frame_count: int,
    fps: float,
    start_ns: int,
) -> list[tuple[int, bytes]]:
    if not source_rows:
        raise ValueError("source_rows is empty")
    timestamps = build_timeline(frame_count, fps, start_ns)
    return [(timestamps[index], source_rows[index % len(source_rows)][1]) for index in range(frame_count)]


def build_joint_rows(
    joint_rows: list[tuple[int, bytes]],
    *,
    output_names: list[str],
    rename_from_base: dict[str, str],
    msgtype: str,
    typestore: object,
    reader: AnyReader,
    noise_std: float = JOINT_NOISE_STD_RAD,
    seed: int = 42,
) -> list[tuple[int, bytes]]:
    index = {name: pos for pos, name in enumerate(BASE_JOINT_NAMES)}
    pick_pairs: list[tuple[str, str]] = []
    for out_name in output_names:
        base_name = rename_from_base.get(out_name, out_name)
        if base_name not in index:
            raise KeyError(f"joint {base_name!r} not found in base bag (for output {out_name!r})")
        pick_pairs.append((out_name, base_name))

    rng = np.random.default_rng(seed)
    joint_cls = type(reader.deserialize(joint_rows[0][1], msgtype))
    filtered: list[tuple[int, bytes]] = []
    for frame_idx, (timestamp, raw) in enumerate(joint_rows):
        msg = reader.deserialize(raw, msgtype)
        names = [out for out, _ in pick_pairs]
        states = np.array(
            [float(msg.joint_states[index[base]]) for _, base in pick_pairs],
            dtype=np.float64,
        )
        if noise_std > 0:
            noise = rng.normal(0.0, noise_std, size=states.shape)
            states = np.clip(states + noise, -3.14, 3.14)
        new_msg = joint_cls(header=msg.header, joint_names=names, joint_states=states)
        filtered.append((timestamp, bytes(typestore.serialize_cdr(new_msg, msgtype))))
    return filtered


def update_meta_for_export(
    meta: dict,
    *,
    cameras: list[dict],
    joint_names: list[str],
    frame_count: int,
    fps: float,
    start_ns: int,
    record_id: str,
    robot_type: str | None = None,
) -> dict:
    out = deepcopy(meta)
    end_ns = start_ns + int(round((frame_count - 1) * 1_000_000_000 / fps)) if frame_count else start_ns
    out["robot_id"] = record_id
    out["robot_type"] = robot_type or out.get("robot_type", "demo_robot")
    out["cameras"] = cameras
    out["joint_names"] = joint_names
    out["start_timestamp_ns"] = start_ns
    out["end_timestamp_ns"] = end_ns
    out["duration_s"] = round((end_ns - start_ns) / 1_000_000_000, 3)
    out["total_frames_count"] = frame_count
    out["is_aligned"] = True
    out["version"] = AF_RECORD_VERSION
    for camera in out["cameras"]:
        camera["fps"] = fps
    if len(joint_names) <= 6:
        out["end_effectors"] = [{"name": "right_hand", "type": "inspire_gripper"}]
    return out


def write_sample(
    name: str,
    meta: dict,
    topic_messages: dict[str, list[tuple[int, bytes]]],
    msgtypes: dict[str, str],
    typestore: object,
) -> None:
    sample_dir = EXAMPLES_DIR / name
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    rosbag_dir = sample_dir / "af_rosbag"
    write_rosbag(rosbag_dir, topic_messages, msgtypes, typestore)
    (sample_dir / "af_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def camera_specs(names: Iterable[str], fps: float) -> list[dict]:
    return [{"name": name, "type": "rgb", "fps": fps} for name in names]


def three_camera_messages(
    messages: dict[str, list[tuple[int, bytes]]],
) -> dict[str, list[tuple[int, bytes]]]:
    return {
        CAMERA_TOPICS[name]: messages[CAMERA_TOPICS[name]]
        for name in THREE_CAM_NAMES
    }


def standard_frame_info(messages: dict[str, list[tuple[int, bytes]]]) -> tuple[int, int, float]:
    head_rows = messages[CAMERA_TOPICS["head_color"]]
    start_ns = head_rows[0][0]
    frame_count = len(head_rows)
    return start_ns, frame_count, 30.0


def build_robot_profile_sample(
    profile: RobotProfile,
    messages: dict[str, list[tuple[int, bytes]]],
    msgtypes: dict[str, str],
    typestore: object,
    meta_template: dict,
    reader: AnyReader,
) -> None:
    start_ns, frame_count, fps = standard_frame_info(messages)
    joint_names = list(profile.joint_names)
    joint_rows = build_joint_rows(
        messages[JOINT_TOPIC],
        output_names=joint_names,
        rename_from_base=dict(profile.base_joint_map),
        msgtype=msgtypes[JOINT_TOPIC],
        typestore=typestore,
        reader=reader,
        seed=sum(ord(ch) for ch in profile.sample_name),
    )
    if len(profile.cameras) == 1:
        topic_messages = {
            CAMERA_TOPICS[profile.cameras[0]]: messages[CAMERA_TOPICS[profile.cameras[0]]],
            JOINT_TOPIC: joint_rows,
        }
    else:
        topic_messages = {**three_camera_messages(messages), JOINT_TOPIC: joint_rows}
    meta = update_meta_for_export(
        meta_template,
        cameras=camera_specs(profile.cameras, fps),
        joint_names=joint_names,
        frame_count=frame_count,
        fps=fps,
        start_ns=start_ns,
        record_id=profile.record_id,
        robot_type=profile.robot_type,
    )
    write_sample(profile.sample_name, meta, topic_messages, msgtypes, typestore)


def build_sample_1cam(
    messages: dict[str, list[tuple[int, bytes]]],
    msgtypes: dict[str, str],
    typestore: object,
    meta_template: dict,
    reader: AnyReader,
) -> None:
    start_ns, frame_count, fps = standard_frame_info(messages)
    joint_names = list(BASE_JOINT_NAMES)
    topic_messages = {
        CAMERA_TOPICS["head_color"]: messages[CAMERA_TOPICS["head_color"]],
        JOINT_TOPIC: messages[JOINT_TOPIC],
    }
    meta = update_meta_for_export(
        meta_template,
        cameras=camera_specs(["head_color"], fps),
        joint_names=joint_names,
        frame_count=frame_count,
        fps=fps,
        start_ns=start_ns,
        record_id="SAMPLE-1CAM",
    )
    write_sample("sample-1cam", meta, topic_messages, msgtypes, typestore)


def build_sample_4cam(
    messages: dict[str, list[tuple[int, bytes]]],
    msgtypes: dict[str, str],
    typestore: object,
    meta_template: dict,
    reader: AnyReader,
) -> None:
    chest_name, chest_topic = EXTRA_CAMERA
    start_ns, frame_count, fps = standard_frame_info(messages)
    head_rows = messages[CAMERA_TOPICS["head_color"]]
    topic_messages = {
        **three_camera_messages(messages),
        chest_topic: [(ts, raw) for ts, raw in head_rows],
        JOINT_TOPIC: messages[JOINT_TOPIC],
    }
    msgtypes = dict(msgtypes)
    msgtypes[chest_topic] = MSG_COMPRESSED_IMAGE
    meta = update_meta_for_export(
        meta_template,
        cameras=camera_specs([*THREE_CAM_NAMES, chest_name], fps),
        joint_names=list(BASE_JOINT_NAMES),
        frame_count=frame_count,
        fps=fps,
        start_ns=start_ns,
        record_id="SAMPLE-4CAM",
    )
    write_sample("sample-4cam", meta, topic_messages, msgtypes, typestore)


def build_sample_meta_mismatch(
    messages: dict[str, list[tuple[int, bytes]]],
    msgtypes: dict[str, str],
    typestore: object,
    meta_template: dict,
    reader: AnyReader,
) -> None:
    start_ns, frame_count, fps = standard_frame_info(messages)
    chest_name, _ = EXTRA_CAMERA
    topic_messages = {**three_camera_messages(messages), JOINT_TOPIC: messages[JOINT_TOPIC]}
    meta = update_meta_for_export(
        meta_template,
        cameras=camera_specs([*THREE_CAM_NAMES, chest_name], fps),
        joint_names=list(BASE_JOINT_NAMES),
        frame_count=frame_count,
        fps=fps,
        start_ns=start_ns,
        record_id="SAMPLE-META-MISMATCH",
    )
    write_sample("sample-meta-mismatch", meta, topic_messages, msgtypes, typestore)


def build_sample_long_5min(
    messages: dict[str, list[tuple[int, bytes]]],
    msgtypes: dict[str, str],
    typestore: object,
    meta_template: dict,
    reader: AnyReader,
) -> None:
    fps = 1.0
    frame_count = 300
    start_ns = messages[CAMERA_TOPICS["head_color"]][0][0]
    topic_messages = {
        CAMERA_TOPICS[name]: resample_messages(messages[CAMERA_TOPICS[name]], frame_count, fps, start_ns)
        for name in THREE_CAM_NAMES
    }
    topic_messages[JOINT_TOPIC] = resample_messages(messages[JOINT_TOPIC], frame_count, fps, start_ns)
    meta = update_meta_for_export(
        meta_template,
        cameras=camera_specs(THREE_CAM_NAMES, fps),
        joint_names=list(BASE_JOINT_NAMES),
        frame_count=frame_count,
        fps=fps,
        start_ns=start_ns,
        record_id="SAMPLE-LONG-5MIN",
    )
    meta["duration_s"] = 300.0
    meta["end_timestamp_ns"] = start_ns + 299_000_000_000
    write_sample("sample-long-5min", meta, topic_messages, msgtypes, typestore)


def build_sample_invalid() -> None:
    sample_dir = EXAMPLES_DIR / "sample-invalid"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)
    bad_meta = {
        "robot_id": "SAMPLE-INVALID",
        "format": "af-record",
    }
    (sample_dir / "af_meta.json").write_text(
        json.dumps(bad_meta, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    (sample_dir / "af_cameras").mkdir()
    readme = """# sample-invalid

故意不符合 af-record RAW 规范，用于测试 `verify_af_raw.py` / `process_af_raw.py` 的错误提示。

问题列表：
- `af_meta.json` 缺少必填字段（cameras、joint_names、total_frames_count 等）
- 缺少 `af_rosbag/` 目录
- 误包含衍生产物目录 `af_cameras/`
"""
    (sample_dir / "README.md").write_text(readme, encoding="utf-8")


GENERATED_SAMPLES = (
    "sample-1cam",
    "sample-4cam",
    "sample-meta-mismatch",
    "sample-long-5min",
    *(profile.sample_name for profile in ROBOT_PROFILES),
    "sample-invalid",
)


def main() -> None:
    if not BASE_SAMPLE.is_dir():
        raise SystemExit(f"Base sample not found: {BASE_SAMPLE}")

    messages, msgtypes, typestore = read_base_messages()
    meta_template = load_meta_template()

    with AnyReader([BASE_SAMPLE / "af_rosbag"]) as reader:
        build_sample_1cam(messages, msgtypes, typestore, meta_template, reader)
        build_sample_4cam(messages, msgtypes, typestore, meta_template, reader)
        build_sample_meta_mismatch(messages, msgtypes, typestore, meta_template, reader)
        build_sample_long_5min(messages, msgtypes, typestore, meta_template, reader)
        for profile in ROBOT_PROFILES:
            build_robot_profile_sample(profile, messages, msgtypes, typestore, meta_template, reader)

    build_sample_invalid()

    stale = EXAMPLES_DIR / "sample-dual-arm-no-torso-3cam"
    if stale.exists():
        shutil.rmtree(stale)

    print("Generated examples (from examples/sample-raw/):")
    print("  - examples/sample-raw/  (base, keep in git)")
    for name in GENERATED_SAMPLES:
        print(f"  - examples/{name}/")


if __name__ == "__main__":
    main()
