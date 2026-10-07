"""Minimal LeRobot dataset v2.1 writer (HF lerobot 0.1.x on-disk layout, no lerobot import)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LEROBOT_CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _append_jsonlines(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _serialize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, dict):
            out[key] = _serialize_stats(value)
        else:
            out[key] = value
    return out


def _feature_stats(array: np.ndarray, *, image: bool = False) -> dict[str, Any]:
    if image:
        arr = array.astype(np.float32) / 255.0
        axes = (0, 2, 3)
        keepdims = True
    else:
        arr = array
        axes = 0
        keepdims = array.ndim == 1
    stats = {
        "min": np.min(arr, axis=axes, keepdims=keepdims),
        "max": np.max(arr, axis=axes, keepdims=keepdims),
        "mean": np.mean(arr, axis=axes, keepdims=keepdims),
        "std": np.std(arr, axis=axes, keepdims=keepdims),
        "count": np.array([len(array)]),
    }
    if image:
        return {k: (v if k == "count" else np.squeeze(v, axis=0)) for k, v in stats.items()}
    return stats


def _init_image_stats(chw: np.ndarray) -> dict[str, Any]:
    """Online image stats from one CHW uint8 frame (pixels treated as samples)."""
    arr = chw.astype(np.float64) / 255.0
    c, h, w = arr.shape
    pixel_count = h * w
    return {
        "min": arr.reshape(c, -1).min(axis=1).reshape(c, 1, 1),
        "max": arr.reshape(c, -1).max(axis=1).reshape(c, 1, 1),
        "sum": arr.reshape(c, -1).sum(axis=1),
        "sumsq": np.square(arr.reshape(c, -1)).sum(axis=1),
        "pixel_count": float(pixel_count),
        "frame_count": 1,
    }


def _update_image_stats(stats: dict[str, Any], chw: np.ndarray) -> dict[str, Any]:
    arr = chw.astype(np.float64) / 255.0
    c = arr.shape[0]
    flat = arr.reshape(c, -1)
    pixel_count = flat.shape[1]
    stats["min"] = np.minimum(stats["min"], flat.min(axis=1).reshape(c, 1, 1))
    stats["max"] = np.maximum(stats["max"], flat.max(axis=1).reshape(c, 1, 1))
    stats["sum"] = stats["sum"] + flat.sum(axis=1)
    stats["sumsq"] = stats["sumsq"] + np.square(flat).sum(axis=1)
    stats["pixel_count"] = stats["pixel_count"] + pixel_count
    stats["frame_count"] = stats["frame_count"] + 1
    return stats


def _finalize_image_stats(stats: dict[str, Any]) -> dict[str, Any]:
    pixel_count = float(stats["pixel_count"])
    mean_flat = stats["sum"] / pixel_count
    var = np.maximum(stats["sumsq"] / pixel_count - np.square(mean_flat), 0.0)
    return {
        "min": stats["min"],
        "max": stats["max"],
        "mean": mean_flat.reshape(-1, 1, 1),
        "std": np.sqrt(var).reshape(-1, 1, 1),
        # Match HF / prior writer: count = number of frames (not pixels).
        "count": np.array([stats["frame_count"]], dtype=np.int64),
    }


def _aggregate_stats(stats_list: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    keys = {key for stats in stats_list for key in stats}
    aggregated: dict[str, dict[str, Any]] = {}
    for key in keys:
        parts = [stats[key] for stats in stats_list if key in stats]
        if not parts:
            continue
        if "count" in parts[0]:
            counts = np.stack([part["count"] for part in parts])
            total_count = counts.sum(axis=0)
            means = np.stack([part["mean"] for part in parts])
            variances = np.stack([part["std"] ** 2 for part in parts])
            while counts.ndim < means.ndim:
                counts = np.expand_dims(counts, axis=-1)
            total_mean = (means * counts).sum(axis=0) / total_count
            delta = means - total_mean
            total_var = ((variances + delta**2) * counts).sum(axis=0) / total_count
            aggregated[key] = {
                "min": np.min(np.stack([part["min"] for part in parts]), axis=0),
                "max": np.max(np.stack([part["max"] for part in parts]), axis=0),
                "mean": total_mean,
                "std": np.sqrt(total_var),
                "count": total_count,
            }
        else:
            aggregated[key] = parts[0]
    return aggregated


def _prepare_features(features: dict[str, Any], use_videos: bool) -> dict[str, Any]:
    prepared = {key: dict(value) for key, value in features.items()}
    for key, ft in prepared.items():
        if ft.get("dtype") == "image" and use_videos:
            ft["dtype"] = "video"
    return {**prepared, **DEFAULT_FEATURES}


def _relative_dataset_path(path: str | Path, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()


def _values_to_pa_array(values: Any, ft: dict[str, Any]) -> Any:
    import pyarrow as pa

    dtype = ft.get("dtype")
    shape = ft.get("shape") or ()

    if dtype == "video":
        video_type = pa.struct([("path", pa.string()), ("timestamp", pa.float32())])
        return pa.array(values, type=video_type)

    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            return pa.array(values)
        if values.ndim == 2:
            if values.shape[1] == 1:
                return pa.array(values.reshape(-1))
            dim = int(shape[0]) if len(shape) == 1 else values.shape[1]
            pa_dtype = pa.float32() if dtype == "float32" else pa.float64()
            if dtype in {"int64", "int32"}:
                pa_dtype = pa.int64() if dtype == "int64" else pa.int32()
            rows = [values[i] for i in range(len(values))]
            return pa.array(rows, type=pa.list_(pa_dtype, dim))

    return pa.array(values)


class LeRobotV21Writer:
    """Write LeRobot v2.1 episodes compatible with HF lerobot 0.1.x datasets."""

    def __init__(
        self,
        root: Path,
        repo_id: str,
        fps: int,
        robot_type: str,
        features: dict[str, Any],
        *,
        use_videos: bool = True,
        ffmpeg: str,
    ) -> None:
        self.root = root
        self.repo_id = repo_id
        self.fps = int(fps)
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.ffmpeg = ffmpeg
        self.features = _prepare_features(features, use_videos)
        self.video_keys = [key for key, ft in self.features.items() if ft.get("dtype") == "video"]
        self.tasks: dict[int, str] = {}
        self.task_to_index: dict[str, int] = {}
        self.episodes: dict[int, dict[str, Any]] = {}
        self.episode_stats: dict[int, dict[str, Any]] = {}
        self.total_frames = 0
        self.episode_buffer: dict[str, Any] | None = None
        self.info = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "fps": self.fps,
            "splits": {},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            if use_videos
            else None,
            "features": self.features,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        _write_json(self.root / "meta" / "info.json", self.info)

    @classmethod
    def create(
        cls,
        *,
        root: str | Path,
        repo_id: str,
        fps: int,
        robot_type: str,
        features: dict[str, Any],
        use_videos: bool = True,
        ffmpeg: str,
        **_ignored: Any,
    ) -> "LeRobotV21Writer":
        return cls(
            root=Path(root),
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            ffmpeg=ffmpeg,
        )

    def _get_task_index(self, task: str) -> int:
        if task in self.task_to_index:
            return self.task_to_index[task]
        task_index = len(self.tasks)
        self.tasks[task_index] = task
        self.task_to_index[task] = task_index
        _append_jsonlines(self.root / "meta" / "tasks.jsonl", {"task_index": task_index, "task": task})
        return task_index

    def _image_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        return (
            self.root
            / "images"
            / image_key
            / f"episode_{episode_index:06d}"
            / f"frame_{frame_index:06d}.png"
        )

    def _data_path(self, episode_index: int) -> Path:
        chunk = episode_index // DEFAULT_CHUNK_SIZE
        return self.root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"

    def _video_path(self, episode_index: int, video_key: str) -> Path:
        chunk = episode_index // DEFAULT_CHUNK_SIZE
        return self.root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    def _create_episode_buffer(self) -> dict[str, Any]:
        buffer = {
            "episode_index": self.info["total_episodes"],
            "size": 0,
            "task": [],
            "frame_index": [],
            "timestamp": [],
            "image_stats": {},
        }
        for key, ft in self.features.items():
            if key in buffer or key in {"index", "episode_index", "task_index"}:
                continue
            buffer[key] = []
        return buffer

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self.episode_buffer is None:
            self.episode_buffer = self._create_episode_buffer()

        payload = dict(frame)
        task = str(payload.pop("task", ""))
        frame_index = self.episode_buffer["size"]
        timestamp = float(payload.pop("timestamp", frame_index / self.fps))

        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        episode_index = self.episode_buffer["episode_index"]
        for key, value in payload.items():
            if key not in self.features:
                raise ValueError(f"Frame key {key!r} is not in dataset features")
            ft = self.features[key]
            if ft["dtype"] in {"image", "video"}:
                img = np.asarray(value, dtype=np.uint8)
                if img.ndim != 3:
                    raise ValueError(f"Expected HxWxC image for {key!r}, got shape {img.shape}")
                path = self._image_path(episode_index, key, frame_index)
                path.parent.mkdir(parents=True, exist_ok=True)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(path), bgr):
                    raise RuntimeError(f"Failed to write image frame: {path}")
                self.episode_buffer[key].append(str(path))
                chw = np.transpose(img, (2, 0, 1))
                running = self.episode_buffer["image_stats"].get(key)
                if running is None:
                    self.episode_buffer["image_stats"][key] = _init_image_stats(chw)
                else:
                    self.episode_buffer["image_stats"][key] = _update_image_stats(running, chw)
            else:
                self.episode_buffer[key].append(np.asarray(value))

        self.episode_buffer["size"] += 1

    def _encode_video(self, image_key: str, episode_index: int) -> str:
        img_dir = self.root / "images" / image_key / f"episode_{episode_index:06d}"
        video_path = self._video_path(episode_index, image_key)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(img_dir / "frame_%06d.png")
        cmd = [
            self.ffmpeg,
            "-y",
            "-framerate",
            str(self.fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = proc.stderr[-2000:] if proc.stderr else "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg video encode failed for {video_path}: {tail}")
        return str(video_path)

    def save_episode(self, episode_data: dict[str, Any] | None = None) -> None:
        if episode_data is None:
            episode_buffer = self.episode_buffer
        else:
            episode_buffer = episode_data
        if not episode_buffer or episode_buffer["size"] < 1:
            raise ValueError("Cannot save empty episode")

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        image_stats = episode_buffer.pop("image_stats", {})
        episode_index = int(episode_buffer["episode_index"])
        episode_tasks = sorted(set(tasks))

        start_index = self.total_frames
        episode_buffer["index"] = np.arange(start_index, start_index + episode_length, dtype=np.int64)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index, dtype=np.int64)
        episode_buffer["task_index"] = np.array(
            [self._get_task_index(task) for task in tasks],
            dtype=np.int64,
        )

        episode_buffer["frame_index"] = np.asarray(episode_buffer["frame_index"], dtype=np.int64)
        episode_buffer["timestamp"] = np.asarray(episode_buffer["timestamp"], dtype=np.float32)

        for key, ft in self.features.items():
            if key in {"index", "episode_index", "task_index", "frame_index", "timestamp"}:
                continue
            if ft["dtype"] in {"image", "video"}:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        table_columns: dict[str, Any] = {}
        for key in self.features:
            if key not in episode_buffer:
                continue
            values = episode_buffer[key]
            if self.features[key]["dtype"] in {"image", "video"}:
                continue
            if isinstance(values, np.ndarray) and values.ndim > 1 and values.shape[1] == 1:
                values = values.reshape(-1)
            table_columns[key] = values

        video_paths: dict[str, str] = {}
        if self.use_videos:
            for key in self.video_keys:
                video_paths[key] = self._encode_video(key, episode_index)
                rel_path = _relative_dataset_path(video_paths[key], self.root)
                table_columns[key] = [
                    {"path": rel_path, "timestamp": float(ts)}
                    for ts in episode_buffer["timestamp"]
                ]

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                'Native LeRobot v2.1 export requires pyarrow: pip install "pyarrow>=14.0.0"'
            ) from exc

        parquet_path = self._data_path(episode_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pa_columns = {
            key: _values_to_pa_array(values, self.features[key])
            for key, values in table_columns.items()
        }
        pq.write_table(pa.table(pa_columns), parquet_path)

        ep_stats: dict[str, Any] = {}
        for key, ft in self.features.items():
            if ft["dtype"] in {"image", "video"}:
                if key in image_stats:
                    ep_stats[key] = _finalize_image_stats(image_stats[key])
            elif key in episode_buffer and isinstance(episode_buffer[key], np.ndarray):
                ep_stats[key] = _feature_stats(episode_buffer[key])

        self.episode_stats[episode_index] = ep_stats
        _append_jsonlines(
            self.root / "meta" / "episodes_stats.jsonl",
            {"episode_index": episode_index, "stats": _serialize_stats(ep_stats)},
        )
        _append_jsonlines(
            self.root / "meta" / "episodes.jsonl",
            {
                "episode_index": episode_index,
                "tasks": episode_tasks,
                "length": episode_length,
            },
        )

        self.episodes[episode_index] = {"length": episode_length, "tasks": episode_tasks}
        self.total_frames += episode_length
        self.info["total_episodes"] += 1
        self.info["total_frames"] = self.total_frames
        self.info["total_tasks"] = len(self.tasks)
        self.info["total_videos"] = self.info["total_episodes"] * len(self.video_keys)
        self.info["total_chunks"] = max(1, math.ceil(self.info["total_episodes"] / DEFAULT_CHUNK_SIZE))
        _write_json(self.root / "meta" / "info.json", self.info)

        images_root = self.root / "images"
        if images_root.exists():
            shutil.rmtree(images_root)

        self.episode_buffer = self._create_episode_buffer()

    def finalize(self) -> None:
        if self.episode_stats:
            _write_json(
                self.root / "meta" / "stats.json",
                _serialize_stats(_aggregate_stats(list(self.episode_stats.values()))),
            )

    def consolidate(self) -> None:
        return
