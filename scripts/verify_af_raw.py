#!/usr/bin/env python3
"""Validate af-record RAW format (af_meta.json + af_rosbag/)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_af_raw import (  # noqa: E402
    ProcessAfRawError,
    _list_image_topics,
    _list_joint_topics,
    assert_joint_msgdef,
    bind_cameras,
    collect_aligned_episode,
    decode_jpeg,
    decode_jpeg_to_bgr,
    extract_joint_states,
    find_joint_topic,
    load_af_meta,
    parse_cameras,
    parse_joint_names,
    parse_timeline_camera,
    parse_total_frames,
    read_topic_messages,
    resolve_export_image_keys,
)
from rosbags.highlevel import AnyReader  # noqa: E402

DERIVED_DIRS = (
    "af_cameras",
    "af_joints",
    "af_mcap",
    "af_annotations",
    "af_lerobot_v2",
    "af_rlds",
)

REQUIRED_META_STRINGS = ("robot_id", "robot_type", "author", "create_time", "format", "version")
REQUIRED_META_NUMBERS = (
    "start_timestamp_ns",
    "end_timestamp_ns",
    "duration_s",
    "total_frames_count",
)
REQUIRED_META_BOOL = ("is_aligned",)
REQUIRED_META_OBJECTS = ("data_validate", "fps_validate", "integrity_validate")
REQUIRED_META_ARRAYS = ("cameras", "end_effectors", "joint_names")


class Level(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CheckResult:
    level: Level
    check: str
    message: str


def _ok(check: str, message: str) -> CheckResult:
    return CheckResult(Level.OK, check, message)


def _warn(check: str, message: str) -> CheckResult:
    return CheckResult(Level.WARN, check, message)


def _err(check: str, message: str) -> CheckResult:
    return CheckResult(Level.ERROR, check, message)


def _validate_bool_object(name: str, value: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(value, dict):
        return [f"{name} must be an object"]
    if "validate" not in value or not isinstance(value["validate"], bool):
        issues.append(f"{name}.validate must be boolean")
    if "reason" not in value or not isinstance(value["reason"], str):
        issues.append(f"{name}.reason must be string")
    return issues


def validate_meta_structure(meta: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    issues: list[str] = []

    for key in REQUIRED_META_STRINGS:
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"missing or empty string field: {key}")

    for key in REQUIRED_META_NUMBERS:
        value = meta.get(key)
        if not isinstance(value, (int, float)):
            issues.append(f"missing or invalid number field: {key}")
        elif key == "total_frames_count" and (not isinstance(value, int) or value < 1):
            issues.append("total_frames_count must be a positive integer")
        elif key in {"start_timestamp_ns", "end_timestamp_ns"} and value <= 0:
            issues.append(f"{key} must be > 0")
        elif key == "duration_s" and value < 0:
            issues.append("duration_s must be >= 0")

    start_ns = meta.get("start_timestamp_ns")
    end_ns = meta.get("end_timestamp_ns")
    if isinstance(start_ns, (int, float)) and isinstance(end_ns, (int, float)) and end_ns < start_ns:
        issues.append("end_timestamp_ns must be >= start_timestamp_ns")

    for key in REQUIRED_META_BOOL:
        if not isinstance(meta.get(key), bool):
            issues.append(f"missing or invalid boolean field: {key}")

    for key in REQUIRED_META_OBJECTS:
        issues.extend(_validate_bool_object(key, meta.get(key)))

    for key in REQUIRED_META_ARRAYS:
        value = meta.get(key)
        if not isinstance(value, list) or not value:
            issues.append(f"missing or empty array field: {key}")

    cameras = meta.get("cameras")
    if isinstance(cameras, list):
        for index, item in enumerate(cameras):
            if not isinstance(item, dict):
                issues.append(f"cameras[{index}] must be an object")
                continue
            for field in ("name", "type"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    issues.append(f"cameras[{index}].{field} must be a non-empty string")
            fps = item.get("fps")
            if not isinstance(fps, (int, float)) or fps <= 0:
                issues.append(f"cameras[{index}].fps must be a positive number")

    end_effectors = meta.get("end_effectors")
    if isinstance(end_effectors, list):
        for index, item in enumerate(end_effectors):
            if not isinstance(item, dict):
                issues.append(f"end_effectors[{index}] must be an object")
                continue
            for field in ("name", "type"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    issues.append(f"end_effectors[{index}].{field} must be a non-empty string")

    joint_names = meta.get("joint_names")
    if isinstance(joint_names, list):
        if not all(isinstance(name, str) and name for name in joint_names):
            issues.append("joint_names must contain only non-empty strings")
        if len(set(joint_names)) != len(joint_names):
            issues.append("joint_names contains duplicates")

    if issues:
        results.append(_err("meta.schema", "; ".join(issues)))
    else:
        results.append(_ok("meta.schema", "af_meta.json contains all required fields"))
    return results


def validate_directory_layout(raw_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    meta_path = raw_dir / "af_meta.json"
    rosbag_dir = raw_dir / "af_rosbag"

    if not raw_dir.is_dir():
        return [_err("layout.root", f"not a directory: {raw_dir}")]
    if not meta_path.is_file():
        results.append(_err("layout.meta", f"missing {meta_path}"))
    if not rosbag_dir.is_dir():
        results.append(_err("layout.rosbag", f"missing {rosbag_dir}"))
    else:
        if not (rosbag_dir / "metadata.yaml").is_file():
            results.append(_err("layout.rosbag", "missing af_rosbag/metadata.yaml"))
        db3_files = list(rosbag_dir.glob("*.db3"))
        if not db3_files:
            results.append(_err("layout.rosbag", "no *.db3 files under af_rosbag/"))
        else:
            results.append(_ok("layout.rosbag", f"found {len(db3_files)} db3 file(s)"))

    present_derived = [name for name in DERIVED_DIRS if (raw_dir / name).exists()]
    if present_derived:
        results.append(
            _warn(
                "layout.derived",
                "RAW directory should not contain derived outputs: "
                + ", ".join(present_derived),
            )
        )
    elif meta_path.is_file() and rosbag_dir.is_dir():
        results.append(_ok("layout.derived", "no derived output directories present"))
    return results


def validate_rosbag_contents(raw_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    rosbag_dir = raw_dir / "af_rosbag"
    meta_path = raw_dir / "af_meta.json"
    if not meta_path.is_file() or not rosbag_dir.is_dir():
        return results

    try:
        meta = load_af_meta(meta_path)
        cameras = parse_cameras(meta)
        joint_names = parse_joint_names(meta)
        timeline_camera = parse_timeline_camera(meta)
        total_frames = parse_total_frames(meta)
    except ProcessAfRawError as exc:
        results.append(_err("rosbag.parse_meta", str(exc)))
        return results

    try:
        with AnyReader([rosbag_dir]) as reader:
            image_topics = _list_image_topics(reader.connections)
            joint_topics = _list_joint_topics(reader.connections)
            results.append(
                _ok(
                    "rosbag.topics",
                    f"{len(image_topics)} CompressedImage topic(s), "
                    f"{len(joint_topics)} joint_states-like topic(s)",
                )
            )

            try:
                bindings = bind_cameras(reader.connections, cameras)
            except ProcessAfRawError as exc:
                results.append(_err("rosbag.camera_bind", str(exc)))
                bindings = []

            if bindings:
                bound_lines = [
                    f"{binding.camera.name!r} -> {binding.connection.topic!r}"
                    for binding in bindings
                ]
                results.append(
                    _ok(
                        "rosbag.camera_bind",
                        f"bound {len(bindings)} camera(s) per af_meta.json: "
                        + "; ".join(bound_lines),
                    )
                )
                used_topics = {binding.connection.topic for binding in bindings}
                unused_topics = sorted(set(image_topics) - used_topics)
                if unused_topics:
                    results.append(
                        _warn(
                            "rosbag.unused_cameras",
                            f"bag has {len(unused_topics)} image topic(s) not listed in "
                            f"af_meta.json (ignored by process_af_raw.py): "
                            + ", ".join(unused_topics),
                        )
                    )
                elif len(image_topics) > len(bindings):
                    results.append(
                        _ok(
                            "rosbag.unused_cameras",
                            "all bag image topics are bound; no extras",
                        )
                    )

            try:
                joint_connection = find_joint_topic(reader.connections)
                assert_joint_msgdef(joint_connection)
                results.append(
                    _ok("rosbag.joint_topic", f"joint topic: {joint_connection.topic!r}")
                )
            except ProcessAfRawError as exc:
                results.append(_err("rosbag.joint_topic", str(exc)))
                joint_connection = None

            if bindings:
                frame_counts: dict[str, int] = {}
                for binding in bindings:
                    count = len(read_topic_messages(reader, binding.connection))
                    frame_counts[binding.camera.name] = count

                count_lines = ", ".join(f"{name}={count}" for name, count in frame_counts.items())
                results.append(_ok("rosbag.frame_counts", count_lines))

                timeline_count = frame_counts.get(timeline_camera)
                if timeline_count is None:
                    results.append(
                        _err(
                            "rosbag.timeline",
                            f"timeline camera {timeline_camera!r} is not among bound cameras",
                        )
                    )
                elif timeline_count != total_frames:
                    results.append(
                        _err(
                            "rosbag.timeline",
                            f"total_frames_count={total_frames} but timeline camera "
                            f"{timeline_camera!r} has {timeline_count} messages",
                        )
                    )
                else:
                    results.append(
                        _ok(
                            "rosbag.timeline",
                            f"timeline camera {timeline_camera!r} matches total_frames_count "
                            f"({total_frames})",
                        )
                    )

                unique_counts = set(frame_counts.values())
                joint_count = None
                if joint_connection is not None:
                    joint_count = len(read_topic_messages(reader, joint_connection))
                    unique_counts.add(joint_count)

                if len(unique_counts) == 1 and joint_count == total_frames:
                    results.append(
                        _ok(
                            "rosbag.alignment",
                            "camera and joint message counts are identical",
                        )
                    )
                else:
                    detail = f"cameras={frame_counts}"
                    if joint_count is not None:
                        detail += f", joints={joint_count}, meta={total_frames}"
                    results.append(_warn("rosbag.alignment", f"frame counts differ: {detail}"))

            if bindings and joint_connection is not None:
                image_key_by_camera = resolve_export_image_keys([b.camera for b in bindings])
                connection_by_camera = {b.camera.name: b.connection for b in bindings}
                by_ts: dict[int, set[str]] = {}
                for camera_name in image_key_by_camera:
                    for timestamp, _ in read_topic_messages(reader, connection_by_camera[camera_name]):
                        by_ts.setdefault(timestamp, set()).add(camera_name)
                for timestamp, _ in read_topic_messages(reader, joint_connection):
                    by_ts.setdefault(timestamp, set()).add("__joint__")

                required = set(image_key_by_camera) | {"__joint__"}
                complete = [ts for ts, present in by_ts.items() if present >= required]
                incomplete = len(by_ts) - len(complete)
                if incomplete:
                    results.append(
                        _err(
                            "rosbag.timestamp_align",
                            f"{incomplete} timestamp(s) missing a meta camera or joint; "
                            f"{len(complete)} fully aligned timestamp(s). "
                            "af_lerobot_v2 / af_rlds export will fail.",
                        )
                    )
                else:
                    results.append(
                        _ok(
                            "rosbag.timestamp_align",
                            f"all {len(complete)} timestamp(s) have every meta camera + joint",
                        )
                    )

            if bindings and joint_connection is not None:
                for binding in bindings:
                    try:
                        _, raw = read_topic_messages(reader, binding.connection)[0]
                        decode_jpeg_to_bgr(decode_jpeg(raw, reader, binding.connection.msgtype))
                        results.append(
                            _ok(
                                "rosbag.decode_image",
                                f"first frame decodes for {binding.camera.name!r}",
                            )
                        )
                    except Exception as exc:
                        results.append(
                            _err(
                                "rosbag.decode_image",
                                f"{binding.camera.name!r}: {type(exc).__name__}: {exc}",
                            )
                        )

                try:
                    _, raw = read_topic_messages(reader, joint_connection)[0]
                    msg = reader.deserialize(raw, joint_connection.msgtype)
                    extract_joint_states(msg, joint_names)
                    results.append(_ok("rosbag.joint_names", "meta joint_names found in bag message"))
                except ProcessAfRawError as exc:
                    results.append(_err("rosbag.joint_names", str(exc)))

            if bindings and joint_connection is not None:
                try:
                    image_key_by_camera = resolve_export_image_keys([b.camera for b in bindings])
                    episode = collect_aligned_episode(
                        reader,
                        bindings,
                        joint_connection,
                        joint_names,
                        image_key_by_camera,
                        task=raw_dir.name,
                    )
                    frame_count = len(episode["states"])
                    results.append(
                        _ok(
                            "rosbag.ml_frames",
                            f"collect_aligned_episode produced {frame_count} frame(s) for ML export",
                        )
                    )
                except ProcessAfRawError as exc:
                    results.append(_err("rosbag.ml_frames", str(exc)))

    except Exception as exc:
        results.append(_err("rosbag.read", f"{type(exc).__name__}: {exc}"))

    return results


def verify_raw(raw_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(validate_directory_layout(raw_dir))

    meta_path = raw_dir / "af_meta.json"
    if meta_path.is_file():
        try:
            meta = load_af_meta(meta_path)
        except ProcessAfRawError as exc:
            results.append(_err("meta.json", str(exc)))
        else:
            results.extend(validate_meta_structure(meta))

    results.extend(validate_rosbag_contents(raw_dir))
    return results


def print_report(raw_dir: Path, results: list[CheckResult]) -> int:
    errors = sum(1 for item in results if item.level == Level.ERROR)
    warns = sum(1 for item in results if item.level == Level.WARN)
    oks = sum(1 for item in results if item.level == Level.OK)

    print(f"af-record RAW verification: {raw_dir}")
    print("-" * 72)
    for item in results:
        print(f"[{item.level.value}] {item.check}: {item.message}")
    print("-" * 72)
    print(f"Summary: {oks} passed, {warns} warning(s), {errors} error(s)")
    if errors:
        print("RESULT: FAIL")
        return 1
    if warns:
        print("RESULT: PASS WITH WARNINGS")
        return 0
    print("RESULT: PASS")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate af-record RAW directory (af_meta.json + af_rosbag/).",
    )
    parser.add_argument(
        "raw_dir",
        type=Path,
        help="Path to af-record RAW input directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    results = verify_raw(raw_dir)
    if args.json:
        payload = {
            "raw_dir": str(raw_dir),
            "results": [
                {"level": item.level.value, "check": item.check, "message": item.message}
                for item in results
            ],
            "summary": {
                "ok": sum(1 for item in results if item.level == Level.OK),
                "warn": sum(1 for item in results if item.level == Level.WARN),
                "error": sum(1 for item in results if item.level == Level.ERROR),
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if payload["summary"]["error"] else 0
    return print_report(raw_dir, results)


if __name__ == "__main__":
    raise SystemExit(main())
