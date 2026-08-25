#!/usr/bin/env python3
"""Parse af-record raw (af_meta.json + af_rosbag) into derived artifacts."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.convert.converter import convert as rosbag_convert
from rosbags.highlevel import AnyReader
from rosbags.interfaces import ConnectionExtRosbag2

MSG_COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
ANNOTATIONS_VERSION = "v0.0.1"
SCHEMA_REL_PATH = Path("schemas") / "annotations.schema.json"
RLDS_DATASET_NAME = "af_record"
RLDS_VERSION = "1.0.0"
# HuggingFace lerobot 0.1.0 (openpi / to_lerobot_v2 compatible). NOT PyPI Cadene lerobot==0.1.0.
LEROBOT_HF_0_1_0_REV = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
LEROBOT_HF_0_1_0_INSTALL = (
    f"pip install \"lerobot @ git+https://github.com/huggingface/lerobot.git@{LEROBOT_HF_0_1_0_REV}\""
)
RESERVED_FRAME_KEYS = frozenset({"state", "action", "task", "image"})


@dataclass(frozen=True)
class CameraSpec:
    name: str
    fps: float


@dataclass
class TopicBinding:
    camera: CameraSpec
    connection: object


class ProcessAfRawError(Exception):
    """Raised when raw record validation or export fails."""


def _list_image_topics(connections) -> list[str]:
    return sorted(
        {
            conn.topic
            for conn in connections
            if is_rosbag2_connection(conn) and conn.msgtype == MSG_COMPRESSED_IMAGE
        }
    )


def _list_joint_topics(connections) -> list[str]:
    return sorted(
        {
            conn.topic
            for conn in connections
            if is_rosbag2_connection(conn) and "joint_states" in conn.topic
        }
    )


def load_af_meta(path: Path) -> dict:
    if not path.is_file():
        raise ProcessAfRawError(f"af_meta.json not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProcessAfRawError(f"af_meta.json is not valid JSON: {exc}") from exc


def parse_timeline_camera(meta: dict) -> str:
    cameras = meta.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ProcessAfRawError("af_meta.json must contain a non-empty cameras list")
    name = cameras[0].get("name")
    if not isinstance(name, str) or not name:
        raise ProcessAfRawError("af_meta.json cameras[0] must have a non-empty name")
    return name


def parse_total_frames(meta: dict) -> int:
    total_frames = meta.get("total_frames_count")
    if not isinstance(total_frames, int) or total_frames < 1:
        raise ProcessAfRawError("af_meta.json must contain total_frames_count >= 1")
    return total_frames


def parse_cameras(meta: dict) -> list[CameraSpec]:
    cameras = meta.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ProcessAfRawError("af_meta.json must contain a non-empty cameras list")
    specs: list[CameraSpec] = []
    for item in cameras:
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ProcessAfRawError(f"Invalid camera entry in af_meta.json: {item!r}")
        fps = float(item.get("fps", 30))
        specs.append(CameraSpec(name=name, fps=fps))
    return specs


def parse_joint_names(meta: dict) -> list[str]:
    joint_names = meta.get("joint_names")
    if not isinstance(joint_names, list) or not joint_names:
        raise ProcessAfRawError("af_meta.json must contain a non-empty joint_names list")
    if not all(isinstance(name, str) and name for name in joint_names):
        raise ProcessAfRawError("joint_names must be a list of non-empty strings")
    return joint_names


def is_rosbag2_connection(conn) -> bool:
    return isinstance(conn.ext, ConnectionExtRosbag2)


def find_camera_topic(connections, camera_name: str):
    matches = [
        conn
        for conn in connections
        if is_rosbag2_connection(conn)
        and conn.msgtype == MSG_COMPRESSED_IMAGE
        and camera_name in conn.topic
    ]
    if not matches:
        available = _list_image_topics(connections)
        raise ProcessAfRawError(
            f"No {MSG_COMPRESSED_IMAGE} topic whose name contains camera {camera_name!r}. "
            f"Available image topics: {available or '(none)'}"
        )
    if len(matches) > 1:
        topics = [conn.topic for conn in matches]
        raise ProcessAfRawError(
            f"Multiple {MSG_COMPRESSED_IMAGE} topics match camera {camera_name!r}: {topics}. "
            "Use a more specific cameras[].name in af_meta.json."
        )
    return matches[0]


def find_joint_topic(connections):
    matches = [
        conn
        for conn in connections
        if is_rosbag2_connection(conn) and "joint_states" in conn.topic
    ]
    if not matches:
        available = _list_joint_topics(connections)
        raise ProcessAfRawError(
            'No topic whose name contains "joint_states". '
            f"Available joint-like topics: {available or '(none)'}"
        )
    if len(matches) > 1:
        topics = [conn.topic for conn in matches]
        raise ProcessAfRawError(
            f'Multiple "joint_states" topics found: {topics}. '
            "Ensure the rosbag contains only one joint feedback topic."
        )
    return matches[0]


def assert_joint_msgdef(connection) -> None:
    msgdef = connection.msgdef.data
    missing = [field for field in ("joint_names", "joint_states") if field not in msgdef]
    if missing:
        raise ProcessAfRawError(
            f"Joint topic {connection.topic!r} msgdef missing fields: {missing}"
        )


def bind_cameras(connections, cameras: list[CameraSpec]) -> list[TopicBinding]:
    bindings: list[TopicBinding] = []
    used_topics: set[str] = set()
    for camera in cameras:
        conn = find_camera_topic(connections, camera.name)
        if conn.topic in used_topics:
            raise ProcessAfRawError(
                f"Camera {camera.name!r} resolves to topic {conn.topic!r}, "
                "which is already bound to another camera. "
                "Use more specific cameras[].name values in af_meta.json."
            )
        used_topics.add(conn.topic)
        bindings.append(TopicBinding(camera=camera, connection=conn))
    return bindings


def read_topic_messages(reader, connection) -> list[tuple[int, bytes]]:
    messages = [
        (timestamp, bytes(raw))
        for conn, timestamp, raw in reader.messages(connections=[connection])
    ]
    if not messages:
        raise ProcessAfRawError(f"Topic {connection.topic!r} contains no messages")
    return messages


def estimate_fps(timestamps: list[int], default: float) -> float:
    if len(timestamps) < 2:
        return default
    deltas = [(timestamps[i + 1] - timestamps[i]) / 1e9 for i in range(len(timestamps) - 1)]
    mean_delta = sum(deltas) / len(deltas)
    if mean_delta <= 0:
        return default
    fps = 1.0 / mean_delta
    return fps if math.isfinite(fps) and fps > 0 else default


def resolve_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise ProcessAfRawError(
            "H.264 encoding requires ffmpeg on PATH or the imageio-ffmpeg package"
        ) from exc


def decode_jpeg(raw_cdr: bytes, reader, msgtype: str) -> bytes:
    msg = reader.deserialize(raw_cdr, msgtype)
    return bytes(msg.data)


def decode_jpeg_to_bgr(jpeg: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ProcessAfRawError("Failed to decode JPEG frame")
    return img


def write_mp4(path: Path, jpeg_frames: list[bytes], fps: float) -> None:
    if not jpeg_frames:
        raise ProcessAfRawError(f"No frames to encode for {path.name}")

    first = decode_jpeg_to_bgr(jpeg_frames[0])
    height, width = first.shape[:2]
    ffmpeg = resolve_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None:
        raise ProcessAfRawError(f"Failed to start ffmpeg for {path.name}")

    def write_frame(img: np.ndarray) -> None:
        if img.shape[0] != height or img.shape[1] != width:
            raise ProcessAfRawError(f"Frame size mismatch for {path.name}")
        proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    try:
        write_frame(first)
        for jpeg in jpeg_frames[1:]:
            write_frame(decode_jpeg_to_bgr(jpeg))
    finally:
        proc.stdin.close()

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        tail = stderr[-2000:] if stderr else "unknown ffmpeg error"
        raise ProcessAfRawError(f"ffmpeg H.264 encode failed for {path.name}: {tail}")


def decode_jpeg_to_rgb(jpeg: bytes) -> np.ndarray:
    bgr = decode_jpeg_to_bgr(jpeg)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resolve_export_image_keys(cameras: list[CameraSpec]) -> dict[str, str]:
    """Map af_meta camera name -> LeRobot / RLDS image feature key."""
    if not cameras:
        raise ProcessAfRawError(
            "Cannot export af_lerobot_v2 / af_rlds: af_meta.json cameras list is empty"
        )

    mapping: dict[str, str] = {}
    used_keys: set[str] = set()
    for index, camera in enumerate(cameras):
        if index == 0:
            feature_key = "image"
        elif camera.name in RESERVED_FRAME_KEYS:
            feature_key = f"camera_{camera.name}"
        else:
            feature_key = camera.name

        if feature_key in used_keys:
            raise ProcessAfRawError(
                f"Duplicate export image key {feature_key!r} for camera {camera.name!r}. "
                "Use distinct cameras[].name values in af_meta.json."
            )
        used_keys.add(feature_key)
        mapping[camera.name] = feature_key
    return mapping


def collect_aligned_episode(
    reader,
    bindings: list[TopicBinding],
    joint_connection,
    joint_names: list[str],
    image_key_by_camera: dict[str, str],
    task: str,
) -> dict[str, Any]:
    """Align streams and keep compressed JPEGs (no full-episode RGB decode)."""
    connection_by_camera = {binding.camera.name: binding.connection for binding in bindings}
    missing_bindings = set(image_key_by_camera) - set(connection_by_camera)
    if missing_bindings:
        raise ProcessAfRawError(
            "Internal error: export image mapping references unknown cameras: "
            f"{sorted(missing_bindings)}"
        )

    by_ts: dict[int, dict[str, bytes]] = {}
    for camera_name in image_key_by_camera:
        connection = connection_by_camera[camera_name]
        for timestamp, raw in read_topic_messages(reader, connection):
            by_ts.setdefault(timestamp, {})[camera_name] = raw

    for timestamp, raw in read_topic_messages(reader, joint_connection):
        by_ts.setdefault(timestamp, {})["__joint__"] = raw

    states: list[np.ndarray] = []
    jpeg_frames: list[dict[str, bytes]] = []
    for timestamp in sorted(by_ts):
        bucket = by_ts[timestamp]
        missing_cameras = set(image_key_by_camera) - set(bucket)
        if missing_cameras or "__joint__" not in bucket:
            details: list[str] = []
            if missing_cameras:
                details.append(
                    "missing cameras: "
                    + ", ".join(
                        f"{name!r} (topic {connection_by_camera[name].topic!r})"
                        for name in sorted(missing_cameras)
                    )
                )
            if "__joint__" not in bucket:
                details.append(f"missing joint topic {joint_connection.topic!r}")
            raise ProcessAfRawError(
                f"Incomplete aligned frame at timestamp {timestamp}: {'; '.join(details)}. "
                "af_lerobot_v2 / af_rlds require identical timestamps across all cameras "
                "listed in af_meta.json and the joint_states topic."
            )
        joint_msg = reader.deserialize(bucket["__joint__"], joint_connection.msgtype)
        state = np.asarray(extract_joint_states(joint_msg, joint_names), dtype=np.float32)
        jpegs: dict[str, bytes] = {}
        for camera_name, feature_key in image_key_by_camera.items():
            jpegs[feature_key] = decode_jpeg(
                bucket[camera_name],
                reader,
                connection_by_camera[camera_name].msgtype,
            )
        states.append(state)
        jpeg_frames.append(jpegs)

    # Free CDR message buffers before decoding any RGB.
    by_ts.clear()

    actions: list[np.ndarray] = []
    for index, state in enumerate(states):
        if index < len(states) - 1:
            actions.append((states[index + 1] - state).astype(np.float32))
        else:
            actions.append(np.zeros_like(state, dtype=np.float32))

    return {
        "task": task,
        "states": states,
        "actions": actions,
        "jpegs": jpeg_frames,
        "image_keys": list(image_key_by_camera.values()),
    }


def iter_decoded_frames(episode: dict[str, Any]):
    """Yield LeRobot-style frames, decoding one RGB set at a time."""
    task = episode["task"]
    for state, action, jpegs in zip(episode["states"], episode["actions"], episode["jpegs"]):
        images = {key: decode_jpeg_to_rgb(jpeg) for key, jpeg in jpegs.items()}
        yield {"state": state, "action": action, "task": task, **images}


def build_aligned_frames(
    reader,
    bindings: list[TopicBinding],
    joint_connection,
    joint_names: list[str],
    image_key_by_camera: dict[str, str],
    task: str,
) -> list[dict[str, Any]]:
    """Compatibility helper: full decode. Prefer collect_aligned_episode for exports."""
    episode = collect_aligned_episode(
        reader,
        bindings,
        joint_connection,
        joint_names,
        image_key_by_camera,
        task,
    )
    return list(iter_decoded_frames(episode))


def _filter_create_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = set(inspect.signature(fn).parameters)
    return {key: value for key, value in kwargs.items() if key in params}


def _post_write_hooks(dataset: Any) -> None:
    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    consolidate = getattr(dataset, "consolidate", None)
    if callable(consolidate):
        consolidate()


def _infer_lerobot_features(
    image_shapes: dict[str, tuple[int, ...]],
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key, shape in image_shapes.items():
        features[key] = {
            "dtype": "image",
            "shape": tuple(shape),
            "names": ["height", "width", "channel"],
        }
    features["state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["action"],
    }
    return features


def _bootstrap_lerobot_vendor_path() -> None:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "third_party" / f"lerobot-{LEROBOT_HF_0_1_0_REV}",
        script_dir / "third_party" / "lerobot",
    ]
    for candidate in candidates:
        module_path = candidate / "lerobot" / "common" / "datasets" / "lerobot_dataset.py"
        if module_path.is_file():
            vendor_path = str(candidate)
            if vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)
            return


def _try_create_lerobot_library_dataset(
    output_dir: Path,
    meta: dict,
    fps: int,
    features: dict[str, Any],
) -> tuple[Any | None, str | None]:
    _bootstrap_lerobot_vendor_path()
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        return None, str(exc)

    robot_id = str(meta.get("robot_id", output_dir.name))
    robot_type = str(meta.get("robot_type", "unknown"))
    create_kw = _filter_create_kwargs(
        LeRobotDataset.create,
        {
            "repo_id": f"local/{robot_id}",
            "robot_type": robot_type,
            "fps": fps,
            "features": features,
            "image_writer_threads": 4,
            "image_writer_processes": 2,
        },
    )

    signature = inspect.signature(LeRobotDataset.create)
    try:
        if "root" in signature.parameters:
            return LeRobotDataset.create(**create_kw, root=str(output_dir)), None

        old_home = os.environ.get("HF_LEROBOT_HOME")
        os.environ["HF_LEROBOT_HOME"] = str(output_dir.parent)
        create_kw["repo_id"] = output_dir.name
        create_kw = _filter_create_kwargs(LeRobotDataset.create, create_kw)
        try:
            return LeRobotDataset.create(**create_kw), None
        finally:
            if old_home is None:
                os.environ.pop("HF_LEROBOT_HOME", None)
            else:
                os.environ["HF_LEROBOT_HOME"] = old_home
    except Exception as exc:
        return None, str(exc)


def _create_lerobot_dataset(output_dir: Path, meta: dict, fps: int, features: dict[str, Any]) -> Any:
    dataset, import_error = _try_create_lerobot_library_dataset(output_dir, meta, fps, features)
    if dataset is not None:
        return dataset

    from lerobot_v21_writer import LeRobotV21Writer

    robot_id = str(meta.get("robot_id", output_dir.name))
    robot_type = str(meta.get("robot_type", "unknown"))
    print(
        "Warning: lerobot.common.datasets.lerobot_dataset unavailable; "
        f"using native LeRobot v2.1 writer ({import_error}). "
        "Optional install from local source: "
        f"third_party/lerobot-{LEROBOT_HF_0_1_0_REV} or {LEROBOT_HF_0_1_0_INSTALL}"
    )
    return LeRobotV21Writer.create(
        root=str(output_dir),
        repo_id=f"local/{robot_id}",
        robot_type=robot_type,
        fps=fps,
        features=features,
        ffmpeg=resolve_ffmpeg(),
    )


def export_lerobot_v2(
    record_root: Path,
    meta: dict,
    reader,
    bindings: list[TopicBinding],
    joint_connection,
    joint_names: list[str],
    out_dir: Path,
    fps: int,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    image_keys = resolve_export_image_keys([binding.camera for binding in bindings])
    episode = collect_aligned_episode(
        reader,
        bindings,
        joint_connection,
        joint_names,
        image_keys,
        task=record_root.name,
    )
    frame_count = len(episode["states"])
    if frame_count < 1:
        raise ProcessAfRawError("No aligned frames available for af_lerobot_v2 export")

    feature_image_keys = [image_keys[binding.camera.name] for binding in bindings]
    first_jpegs = episode["jpegs"][0]
    image_shapes = {
        key: tuple(decode_jpeg_to_rgb(first_jpegs[key]).shape) for key in feature_image_keys
    }
    state_dim = int(np.asarray(episode["states"][0]).shape[0])
    action_dim = int(np.asarray(episode["actions"][0]).shape[0])
    dataset = _create_lerobot_dataset(
        out_dir,
        meta,
        fps,
        _infer_lerobot_features(image_shapes, state_dim, action_dim),
    )
    for frame in iter_decoded_frames(episode):
        dataset.add_frame(frame)
    dataset.save_episode()
    _post_write_hooks(dataset)
    print(f"Wrote af_lerobot_v2 dataset to {out_dir} ({frame_count} frames)")


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ProcessAfRawError(
            'af_rlds export requires tensorflow: pip install "tensorflow>=2.13.0"'
        ) from exc
    return tf


def _rlds_bytes_feature(tf: Any, value: bytes) -> Any:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _rlds_float_feature(tf: Any, values: list[float]) -> Any:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _rlds_int64_feature(tf: Any, value: int) -> Any:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _build_rlds_features_schema(
    image_shapes: dict[str, tuple[int, ...]],
    state_dim: int,
    action_dim: int,
    joint_names: list[str],
    camera_names: list[str],
    image_keys: list[str],
) -> dict[str, Any]:
    step_features: dict[str, Any] = {
        "steps/observation/state": {"dtype": "float32", "shape": [state_dim]},
        "steps/action": {"dtype": "float32", "shape": [action_dim]},
        "steps/discount": {"dtype": "float32", "shape": []},
        "steps/reward": {"dtype": "float32", "shape": []},
        "steps/is_first": {"dtype": "int64", "shape": []},
        "steps/is_last": {"dtype": "int64", "shape": []},
        "steps/is_terminal": {"dtype": "int64", "shape": []},
        "steps/language_instruction": {"dtype": "string", "shape": []},
    }
    for key in image_keys:
        step_features[f"steps/observation/{key}"] = {
            "dtype": "jpeg_bytes",
            "shape": list(image_shapes[key]),
        }
    return {
        "format": "af_rlds",
        "serialization": "tf.train.SequenceExample",
        "episode_context": {
            "episode/recording_id": "string",
            "episode/robot_id": "string",
            "episode/robot_type": "string",
            "episode/fps": "int64",
            "episode/joint_names": "json_string",
            "episode/camera_names": "json_string",
            "episode/camera_feature_keys": "json_string",
        },
        "steps_feature_lists": step_features,
        "joint_names": joint_names,
        "camera_names": camera_names,
        "camera_feature_keys": image_keys,
    }


def _encode_rlds_sequence_example(
    tf: Any,
    episode: dict[str, Any],
    image_keys: list[str],
    meta: dict,
    record_name: str,
    fps: int,
    joint_names: list[str],
    camera_names: list[str],
) -> bytes:
    states = episode["states"]
    actions = episode["actions"]
    jpegs = episode["jpegs"]
    task = str(episode["task"])
    last_index = len(states) - 1

    context = tf.train.Features(
        feature={
            "episode/recording_id": _rlds_bytes_feature(tf, record_name.encode("utf-8")),
            "episode/robot_id": _rlds_bytes_feature(
                tf, str(meta.get("robot_id", record_name)).encode("utf-8")
            ),
            "episode/robot_type": _rlds_bytes_feature(
                tf, str(meta.get("robot_type", "unknown")).encode("utf-8")
            ),
            "episode/fps": _rlds_int64_feature(tf, int(fps)),
            "episode/joint_names": _rlds_bytes_feature(
                tf, json.dumps(list(joint_names), ensure_ascii=False).encode("utf-8")
            ),
            "episode/camera_names": _rlds_bytes_feature(
                tf, json.dumps(list(camera_names), ensure_ascii=False).encode("utf-8")
            ),
            "episode/camera_feature_keys": _rlds_bytes_feature(
                tf, json.dumps(list(image_keys), ensure_ascii=False).encode("utf-8")
            ),
        }
    )

    feature_lists: dict[str, Any] = {}
    for key in image_keys:
        feature_lists[f"steps/observation/{key}"] = tf.train.FeatureList(
            feature=[_rlds_bytes_feature(tf, frame_jpegs[key]) for frame_jpegs in jpegs]
        )

    feature_lists["steps/observation/state"] = tf.train.FeatureList(
        feature=[
            _rlds_float_feature(tf, np.asarray(state, dtype=np.float32).tolist())
            for state in states
        ]
    )
    feature_lists["steps/action"] = tf.train.FeatureList(
        feature=[
            _rlds_float_feature(tf, np.asarray(action, dtype=np.float32).tolist())
            for action in actions
        ]
    )
    feature_lists["steps/discount"] = tf.train.FeatureList(
        feature=[_rlds_float_feature(tf, [1.0]) for _ in states]
    )
    feature_lists["steps/reward"] = tf.train.FeatureList(
        feature=[_rlds_float_feature(tf, [0.0]) for _ in states]
    )
    feature_lists["steps/is_first"] = tf.train.FeatureList(
        feature=[_rlds_int64_feature(tf, int(index == 0)) for index in range(len(states))]
    )
    feature_lists["steps/is_last"] = tf.train.FeatureList(
        feature=[
            _rlds_int64_feature(tf, int(index == last_index)) for index in range(len(states))
        ]
    )
    feature_lists["steps/is_terminal"] = tf.train.FeatureList(
        feature=[
            _rlds_int64_feature(tf, int(index == last_index)) for index in range(len(states))
        ]
    )
    feature_lists["steps/language_instruction"] = tf.train.FeatureList(
        feature=[_rlds_bytes_feature(tf, task.encode("utf-8")) for _ in states]
    )

    sequence = tf.train.SequenceExample(
        context=context,
        feature_lists=tf.train.FeatureLists(feature_list=feature_lists),
    )
    return sequence.SerializeToString()


def export_rlds(
    record_root: Path,
    meta: dict,
    reader,
    bindings: list[TopicBinding],
    joint_connection,
    joint_names: list[str],
    out_dir: Path,
    fps: int,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    tf = _import_tensorflow()
    image_key_by_camera = resolve_export_image_keys([binding.camera for binding in bindings])
    feature_image_keys = [image_key_by_camera[binding.camera.name] for binding in bindings]
    camera_names = [binding.camera.name for binding in bindings]
    episode = collect_aligned_episode(
        reader,
        bindings,
        joint_connection,
        joint_names,
        image_key_by_camera,
        task=record_root.name,
    )
    frame_count = len(episode["states"])
    if frame_count < 1:
        raise ProcessAfRawError("No aligned frames available for af_rlds export")

    first_jpegs = episode["jpegs"][0]
    image_shapes = {
        key: tuple(decode_jpeg_to_rgb(first_jpegs[key]).shape) for key in feature_image_keys
    }
    state_dim = int(np.asarray(episode["states"][0]).shape[0])
    action_dim = int(np.asarray(episode["actions"][0]).shape[0])
    features_schema = _build_rlds_features_schema(
        image_shapes,
        state_dim,
        action_dim,
        joint_names,
        camera_names,
        feature_image_keys,
    )

    version_dir = out_dir / RLDS_VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "features.json").write_text(
        json.dumps(features_schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    shard_name = f"{RLDS_DATASET_NAME}-train.tfrecord-00000-of-00001"
    tfrecord_path = version_dir / shard_name
    encoded = _encode_rlds_sequence_example(
        tf,
        episode,
        feature_image_keys,
        meta,
        record_root.name,
        fps,
        joint_names,
        camera_names,
    )
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        writer.write(encoded)

    num_bytes = tfrecord_path.stat().st_size
    dataset_info = {
        "description": "af-record raw rosbag converted to RLDS TFRecord (SequenceExample)",
        "citation": "",
        "sizeInBytes": str(num_bytes),
        "downloadSize": "0",
        "fileFormat": "tfrecord",
        "moduleName": "process_af_raw",
        "name": RLDS_DATASET_NAME,
        "releaseNotes": {},
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": str(num_bytes),
                "shardLengths": [str(frame_count)],
            }
        ],
        "version": RLDS_VERSION,
    }
    (version_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote af_rlds dataset to {out_dir} ({frame_count} frames)")


def extract_joint_states(msg, expected_names: list[str]) -> list[float]:
    if not hasattr(msg, "joint_names") or not hasattr(msg, "joint_states"):
        raise ProcessAfRawError("Joint message does not expose joint_names / joint_states")
    names = [str(name) for name in msg.joint_names]
    values = [float(value) for value in msg.joint_states]
    if len(names) != len(values):
        raise ProcessAfRawError(
            f"joint_names length ({len(names)}) != joint_states length ({len(values)})"
        )
    if names != expected_names:
        index = {name: idx for idx, name in enumerate(names)}
        missing = [name for name in expected_names if name not in index]
        if missing:
            raise ProcessAfRawError(
                f"af_meta joint_names missing from bag joint topic: {missing}"
            )
        values = [values[index[name]] for name in expected_names]
    return values


def export_cameras(reader, bindings: list[TopicBinding], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_counts: dict[str, int] = {}
    for binding in bindings:
        messages = read_topic_messages(reader, binding.connection)
        timestamps = [ts for ts, _ in messages]
        fps = estimate_fps(timestamps, binding.camera.fps)
        jpeg_frames = [
            decode_jpeg(raw, reader, binding.connection.msgtype) for _, raw in messages
        ]
        out_path = out_dir / f"{binding.camera.name}.mp4"
        write_mp4(out_path, jpeg_frames, fps)
        frame_counts[binding.camera.name] = len(messages)
        print(f"Wrote {out_path} ({len(messages)} frames @ {fps:.3f} fps)")
    return frame_counts


def export_joints(
    reader,
    joint_connection,
    expected_names: list[str],
    out_dir: Path,
) -> int:
    messages = read_topic_messages(reader, joint_connection)
    states: list[list[float]] = []
    for _, raw in messages:
        msg = reader.deserialize(raw, joint_connection.msgtype)
        states.append(extract_joint_states(msg, expected_names))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"joint_names": expected_names, "states": states}
    out_path = out_dir / "joint_states.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({len(states)} frames)")
    return len(states)


def export_mcap(raw_rosbag_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rosbag_convert(
        srcs=[raw_rosbag_dir],
        dst=out_dir,
        dst_storage="mcap",
        dst_version=9,
        compress=None,
        compress_mode="none",
        default_typestore=None,
        typestore=None,
        exclude_topics=[],
        include_topics=[],
        exclude_msgtypes=[],
        include_msgtypes=[],
    )
    print(f"Wrote MCAP rosbag to {out_dir}")


def empty_annotations(timeline_camera: str, total_frames: int) -> dict:
    return {
        "format": "af-annotations",
        "version": ANNOTATIONS_VERSION,
        "reference": {
            "timeline_camera": timeline_camera,
            "total_frames": total_frames,
            "frame_index_origin": 0,
            "interval": "half_open",
        },
        "main": [],
        "region_marker": {
            "success_markers": [],
            "interference_markers": [],
            "error_markers": [],
            "single_frame_markers": [],
        },
    }


def init_annotations(out_dir: Path, timeline_camera: str, total_frames: int, schema_src: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(empty_annotations(timeline_camera, total_frames), ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    if not schema_src.is_file():
        raise ProcessAfRawError(f"annotations schema not found: {schema_src}")
    shutil.copy2(schema_src, out_dir / "annotations.schema.json")
    print(f"Initialized {out_dir}")


def update_meta_in_place(meta_path: Path, is_aligned: bool) -> None:
    meta = load_af_meta(meta_path)
    meta["is_aligned"] = is_aligned
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"Updated {meta_path}")


def prepare_output_dir(output_root: Path, *, inplace: bool = False) -> None:
    names = (
        "af_cameras",
        "af_joints",
        "af_mcap",
        "af_annotations",
        "af_lerobot_v2",
        "af_rlds",
    )
    if not inplace:
        names = (*names, "af_rosbag")
    for name in names:
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)


def ensure_output_meta(input_root: Path, output_root: Path) -> Path:
    input_meta = input_root / "af_meta.json"
    output_meta = output_root / "af_meta.json"
    if input_meta.resolve() == output_meta.resolve():
        return output_meta
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_meta, output_meta)
    return output_meta


def ensure_output_rosbag(input_root: Path, output_root: Path) -> Path:
    input_rosbag = input_root / "af_rosbag"
    output_rosbag = output_root / "af_rosbag"
    if not input_rosbag.is_dir():
        raise ProcessAfRawError(f"af_rosbag directory not found: {input_rosbag}")
    if input_rosbag.resolve() == output_rosbag.resolve():
        return output_rosbag
    output_root.mkdir(parents=True, exist_ok=True)
    if output_rosbag.exists():
        shutil.rmtree(output_rosbag)
    shutil.copytree(input_rosbag, output_rosbag)
    print(f"Copied af_rosbag to {output_rosbag}")
    return output_rosbag


def process_raw_record(input_root: Path, output_root: Path, schema_src: Path) -> None:
    input_meta_path = input_root / "af_meta.json"
    rosbag_dir = input_root / "af_rosbag"
    if not rosbag_dir.is_dir():
        raise ProcessAfRawError(f"af_rosbag directory not found: {rosbag_dir}")

    meta = load_af_meta(input_meta_path)
    cameras = parse_cameras(meta)
    joint_names = parse_joint_names(meta)
    timeline_camera = parse_timeline_camera(meta)
    total_frames = parse_total_frames(meta)

    output_root.mkdir(parents=True, exist_ok=True)
    inplace = input_root.resolve() == output_root.resolve()
    output_meta_path = ensure_output_meta(input_root, output_root)
    prepare_output_dir(output_root, inplace=inplace)
    ensure_output_rosbag(input_root, output_root)

    lerobot_fps = int(round(cameras[0].fps))
    with AnyReader([rosbag_dir]) as reader:
        bindings = bind_cameras(reader.connections, cameras)
        bound_topics = {binding.connection.topic for binding in bindings}
        unused_image_topics = sorted(set(_list_image_topics(reader.connections)) - bound_topics)
        if unused_image_topics:
            print(
                "Note: ignoring bag image topic(s) not listed in af_meta.json cameras[]: "
                + ", ".join(unused_image_topics)
            )
        joint_connection = find_joint_topic(reader.connections)
        assert_joint_msgdef(joint_connection)

        camera_frame_counts = export_cameras(reader, bindings, output_root / "af_cameras")
        joint_frame_count = export_joints(
            reader,
            joint_connection,
            joint_names,
            output_root / "af_joints",
        )
        export_lerobot_v2(
            input_root,
            meta,
            reader,
            bindings,
            joint_connection,
            joint_names,
            output_root / "af_lerobot_v2",
            lerobot_fps,
        )
        export_rlds(
            input_root,
            meta,
            reader,
            bindings,
            joint_connection,
            joint_names,
            output_root / "af_rlds",
            lerobot_fps,
        )

    export_mcap(rosbag_dir, output_root / "af_mcap")

    timeline_frames = camera_frame_counts.get(timeline_camera)
    if timeline_frames is None:
        raise ProcessAfRawError(f"Timeline camera {timeline_camera!r} was not exported")
    if timeline_frames != total_frames:
        raise ProcessAfRawError(
            f"af_meta total_frames_count ({total_frames}) does not match "
            f"timeline camera {timeline_camera!r} topic frames ({timeline_frames}). "
            f"Update total_frames_count or ensure cameras[0] matches the reference stream."
        )

    all_counts = set(camera_frame_counts.values()) | {joint_frame_count}
    is_aligned = len(all_counts) == 1 and joint_frame_count == total_frames
    if not is_aligned:
        print(
            "Warning: frame counts differ across streams: "
            f"cameras={camera_frame_counts}, joints={joint_frame_count}, "
            f"meta={total_frames}"
        )

    init_annotations(
        output_root / "af_annotations",
        timeline_camera=timeline_camera,
        total_frames=total_frames,
        schema_src=schema_src,
    )
    update_meta_in_place(output_meta_path, is_aligned)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export derived artifacts from af-record raw (af_meta.json + af_rosbag).",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input record directory containing af_meta.json and af_rosbag/",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as input_dir, in-place export)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_root = args.input_dir.resolve()
    output_root = args.output.resolve() if args.output is not None else input_root
    schema_src = Path(__file__).resolve().parent / SCHEMA_REL_PATH

    process_raw_record(input_root, output_root, schema_src)


if __name__ == "__main__":
    try:
        main()
    except ProcessAfRawError as exc:
        raise SystemExit(f"[process_af_raw] {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("[process_af_raw] interrupted") from None
    except Exception as exc:
        raise SystemExit(
            f"[process_af_raw] unexpected {type(exc).__name__}: {exc}"
        ) from exc
