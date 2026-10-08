"""Minimal LeRobot dataset v3.0 writer (HF lerobot >=0.4 on-disk layout, no lerobot import).

Layout matches official LeRobotDataset v3.0:
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-XXX/file-YYY.parquet
  data/chunk-XXX/file-YYY.parquet
  videos/{video_key}/chunk-XXX/file-YYY.mp4

This toolkit typically writes one episode per record; file-based sharding still uses
the v3 path templates (file-000) so loaders see a valid single-shard dataset.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot_v21_writer import (
    _finalize_image_stats,
    _init_image_stats,
    _serialize_stats,
    _update_image_stats,
    _feature_stats,
)

LEROBOT_CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"

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


def _flatten_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested episode stats to parquet columns (stats/feat/min, ...)."""
    out: dict[str, Any] = {}
    for key, value in stats.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict) and not any(
            k in value for k in ("min", "max", "mean", "std", "count")
        ):
            out.update(_flatten_stats(path, value))
            continue
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                out[f"{path}/{metric}"] = (
                    metric_value.tolist()
                    if isinstance(metric_value, np.ndarray)
                    else metric_value
                )
        elif isinstance(value, np.ndarray):
            out[path] = value.tolist()
        else:
            out[path] = value
    return out


def _prepare_features(
    features: dict[str, Any],
    *,
    fps: int,
    use_videos: bool,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for key, value in features.items():
        ft = dict(value)
        shape = ft.get("shape")
        if isinstance(shape, list):
            ft["shape"] = tuple(shape)
        if ft.get("dtype") == "image" and use_videos:
            ft["dtype"] = "video"
            height, width, channels = (int(x) for x in ft["shape"][:3])
            ft["info"] = {
                "video.height": height,
                "video.width": width,
                "video.channels": channels,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": int(fps),
                "has_audio": False,
            }
        if ft.get("dtype") != "video":
            ft["fps"] = int(fps)
        prepared[key] = ft

    for key, value in DEFAULT_FEATURES.items():
        ft = dict(value)
        ft["fps"] = int(fps)
        prepared[key] = ft
    return prepared


def _values_to_pa_array(values: Any, ft: dict[str, Any]) -> Any:
    import pyarrow as pa

    dtype = ft.get("dtype")
    shape = ft.get("shape") or ()

    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            return pa.array(values)
        if values.ndim == 2:
            if values.shape[1] == 1:
                return pa.array(values.reshape(-1))
            dim = int(shape[0]) if len(shape) == 1 else values.shape[1]
            if dtype == "float32":
                pa_dtype = pa.float32()
            elif dtype == "float64":
                pa_dtype = pa.float64()
            elif dtype == "int64":
                pa_dtype = pa.int64()
            elif dtype == "int32":
                pa_dtype = pa.int32()
            else:
                pa_dtype = pa.float32()
            rows = [values[i] for i in range(len(values))]
            return pa.array(rows, type=pa.list_(pa_dtype, dim))

    return pa.array(values)


class LeRobotV30Writer:
    """Write LeRobot v3.0 episodes compatible with HF lerobot >= 0.4 datasets."""

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
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.ffmpeg = ffmpeg
        self.features = _prepare_features(features, fps=self.fps, use_videos=use_videos)
        self.video_keys = [key for key, ft in self.features.items() if ft.get("dtype") == "video"]
        self.tasks: dict[int, str] = {}
        self.task_to_index: dict[str, int] = {}
        self.episode_stats: dict[int, dict[str, Any]] = {}
        self.episode_rows: list[dict[str, Any]] = []
        self.total_frames = 0
        self.episode_buffer: dict[str, Any] | None = None
        self._finalized = False

        self.info: dict[str, Any] = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
            "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            "fps": self.fps,
            "splits": {},
            "data_path": DEFAULT_DATA_PATH,
            "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
            "features": {
                key: {
                    **{k: (list(v) if k == "shape" and isinstance(v, tuple) else v) for k, v in ft.items()}
                }
                for key, ft in self.features.items()
            },
        }
        if self.root.exists():
            shutil.rmtree(self.root)
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
    ) -> "LeRobotV30Writer":
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
        return task_index

    def _image_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        return (
            self.root
            / "images"
            / image_key
            / f"episode-{episode_index:06d}"
            / f"frame-{frame_index:06d}.png"
        )

    def _data_path(self, chunk_index: int, file_index: int) -> Path:
        return self.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)

    def _video_path(self, video_key: str, chunk_index: int, file_index: int) -> Path:
        return self.root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )

    def _episodes_path(self, chunk_index: int, file_index: int) -> Path:
        return self.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )

    def _create_episode_buffer(self) -> dict[str, Any]:
        buffer: dict[str, Any] = {
            "episode_index": self.info["total_episodes"],
            "size": 0,
            "task": [],
            "frame_index": [],
            "timestamp": [],
            "image_stats": {},
        }
        for key in self.features:
            if key in buffer or key in {"index", "episode_index", "task_index"}:
                continue
            buffer[key] = []
        return buffer

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add_frame after finalize()")
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

    def _encode_video(self, image_key: str, episode_index: int, chunk_index: int, file_index: int) -> Path:
        img_dir = self.root / "images" / image_key / f"episode-{episode_index:06d}"
        video_path = self._video_path(image_key, chunk_index, file_index)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(img_dir / "frame-%06d.png")
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
        return video_path

    def save_episode(self, episode_data: dict[str, Any] | None = None) -> None:
        if self._finalized:
            raise RuntimeError("Cannot save_episode after finalize()")
        # af-record exports one recording = one episode into a single v3 shard.
        if self.info["total_episodes"] > 0:
            raise NotImplementedError(
                "lerobot_v30_writer currently supports one episode per dataset "
                "(matches af-record single-recording export)"
            )
        episode_buffer = self.episode_buffer if episode_data is None else episode_data
        if not episode_buffer or episode_buffer["size"] < 1:
            raise ValueError("Cannot save empty episode")

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                'Native LeRobot v3.0 export requires pyarrow: pip install "pyarrow>=14.0.0"'
            ) from exc

        episode_length = int(episode_buffer.pop("size"))
        tasks = episode_buffer.pop("task")
        image_stats = episode_buffer.pop("image_stats", {})
        episode_index = int(episode_buffer["episode_index"])
        episode_tasks = sorted(set(tasks))
        for task in episode_tasks:
            self._get_task_index(task)

        # Single-episode / append-friendly shard indices for this toolkit.
        data_chunk_index = 0
        data_file_index = 0
        video_chunk_index = 0
        video_file_index = 0

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

        # Parquet holds only non-visual columns; videos are located via episode metadata.
        table_columns: dict[str, Any] = {}
        for key, ft in self.features.items():
            if ft["dtype"] in {"image", "video"}:
                continue
            values = episode_buffer[key]
            if isinstance(values, np.ndarray) and values.ndim > 1 and values.shape[1] == 1:
                values = values.reshape(-1)
            table_columns[key] = values

        if self.use_videos:
            for key in self.video_keys:
                self._encode_video(key, episode_index, video_chunk_index, video_file_index)

        parquet_path = self._data_path(data_chunk_index, data_file_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pa_columns = {
            key: _values_to_pa_array(values, self.features[key])
            for key, values in table_columns.items()
        }
        table = pa.table(pa_columns)
        # One row-group per episode keeps shards random-access friendly.
        with pq.ParquetWriter(str(parquet_path), table.schema, compression="snappy") as writer:
            writer.write_table(table)

        ep_stats: dict[str, Any] = {}
        for key, ft in self.features.items():
            if ft["dtype"] in {"image", "video"}:
                if key in image_stats:
                    ep_stats[key] = _finalize_image_stats(image_stats[key])
            elif key in episode_buffer and isinstance(episode_buffer[key], np.ndarray):
                ep_stats[key] = _feature_stats(episode_buffer[key])
        self.episode_stats[episode_index] = ep_stats

        duration_s = float(episode_length) / float(self.fps)
        ep_row: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            "data/chunk_index": data_chunk_index,
            "data/file_index": data_file_index,
            "dataset_from_index": start_index,
            "dataset_to_index": start_index + episode_length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            **_flatten_stats("stats", ep_stats),
        }
        if self.use_videos:
            for key in self.video_keys:
                ep_row[f"videos/{key}/chunk_index"] = video_chunk_index
                ep_row[f"videos/{key}/file_index"] = video_file_index
                ep_row[f"videos/{key}/from_timestamp"] = 0.0
                ep_row[f"videos/{key}/to_timestamp"] = duration_s
        self.episode_rows.append(ep_row)

        self.total_frames += episode_length
        self.info["total_episodes"] += 1
        self.info["total_frames"] = self.total_frames
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        _write_json(self.root / "meta" / "info.json", self.info)

        images_root = self.root / "images"
        if images_root.exists():
            shutil.rmtree(images_root)

        self.episode_buffer = self._create_episode_buffer()

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.info["total_episodes"] < 1:
            raise RuntimeError("Cannot finalize empty LeRobot v3.0 dataset")

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                'Native LeRobot v3.0 export requires pyarrow: pip install "pyarrow>=14.0.0"'
            ) from exc

        # tasks.parquet: index = task string, column = task_index
        task_strings = [self.tasks[idx] for idx in sorted(self.tasks)]
        task_indices = sorted(self.tasks)
        tasks_table = pa.table(
            {
                "task": task_strings,
                "task_index": np.asarray(task_indices, dtype=np.int64),
            }
        )
        tasks_path = self.root / DEFAULT_TASKS_PATH
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tasks_table, tasks_path)

        # Episode metadata parquet (single shard for this toolkit).
        episodes_path = self._episodes_path(0, 0)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        # Build column-oriented table; list-valued "tasks" needs explicit type.
        columns: dict[str, Any] = {}
        keys = list(self.episode_rows[0].keys())
        for key in keys:
            values = [row[key] for row in self.episode_rows]
            if key == "tasks":
                columns[key] = pa.array(values, type=pa.list_(pa.string()))
            else:
                columns[key] = values
        pq.write_table(pa.table(columns), episodes_path)

        # Global stats.json (episode aggregation for single-episode is identity).
        if self.episode_stats:
            # For one or more episodes, prefer simple mean aggregation of feature stats.
            aggregated: dict[str, Any] = {}
            feature_keys = set()
            for stats in self.episode_stats.values():
                feature_keys.update(stats.keys())
            for feature_key in feature_keys:
                parts = [stats[feature_key] for stats in self.episode_stats.values() if feature_key in stats]
                if not parts:
                    continue
                if len(parts) == 1:
                    aggregated[feature_key] = parts[0]
                    continue
                counts = np.stack([part["count"] for part in parts])
                total_count = counts.sum(axis=0)
                means = np.stack([part["mean"] for part in parts])
                variances = np.stack([part["std"] ** 2 for part in parts])
                while counts.ndim < means.ndim:
                    counts = np.expand_dims(counts, axis=-1)
                total_mean = (means * counts).sum(axis=0) / total_count
                delta = means - total_mean
                total_var = ((variances + delta**2) * counts).sum(axis=0) / total_count
                aggregated[feature_key] = {
                    "min": np.min(np.stack([part["min"] for part in parts]), axis=0),
                    "max": np.max(np.stack([part["max"] for part in parts]), axis=0),
                    "mean": total_mean,
                    "std": np.sqrt(total_var),
                    "count": total_count,
                }
            _write_json(self.root / "meta" / "stats.json", _serialize_stats(aggregated))

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        _write_json(self.root / "meta" / "info.json", self.info)
        self._finalized = True

    def consolidate(self) -> None:
        return
