#!/usr/bin/env python3
"""Template: migrate af-record v1.0.1 in MinIO → add af_lerobot_v3, bump format to v1.0.2.

设计意图
--------
- 从本地 MySQL 读取记录（主要是对象存储路径）
- 在 MinIO 中定位 format version == v1.0.1 的 af-record 目录
- 补齐缺失的 ``af_lerobot_v3/``，并把 ``af_meta.json`` 的 ``version`` 改为 ``v1.0.2``

本文件是**模板骨架**：全局配置、SQL、表字段名需按你们环境填写；
标有 ``NotImplementedError`` / ``TODO`` 的方法可按实际存储布局实现。

依赖
----
已写入仓库 ``requirements.txt``（``pymysql``、``minio``）。安装：

    pip install -r requirements.txt

建议用法
--------
    # 1. 填好下方 MYSQL_* / MINIO_* / SQL
    # 2. 实现 TODO 方法
    # 3. 先 dry-run：
    python scripts/migrate_af_record_add_lerobot_v3.py --dry-run
    # 4. 再正式跑：
    python scripts/migrate_af_record_add_lerobot_v3.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------------------
# 全局配置（请按环境填写；模板中故意留空 / 占位）
# ---------------------------------------------------------------------------

# MySQL
MYSQL_HOST: str = ""
MYSQL_PORT: int = 3306
MYSQL_USER: str = ""
MYSQL_PASSWORD: str = ""
MYSQL_DATABASE: str = ""
MYSQL_CHARSET: str = "utf8mb4"

# 查询「需要迁移」的记录。请改成你们真实表/字段。
# 期望至少选出：record_id、对象存储前缀（或 bucket+key）、当前 format version。
MYSQL_QUERY_PENDING: str = """
SELECT
    id              AS record_id,
    object_prefix   AS object_prefix,
    format_version  AS format_version
FROM af_record
WHERE format_version = 'v1.0.1'
  AND deleted = 0
ORDER BY id ASC
LIMIT 100
"""

# MinIO / S3 兼容
MINIO_ENDPOINT: str = ""          # e.g. "127.0.0.1:9000"
MINIO_ACCESS_KEY: str = ""
MINIO_SECRET_KEY: str = ""
MINIO_SECURE: bool = False
MINIO_BUCKET: str = ""            # 若每条记录自带 bucket，可在 Record 里覆盖

# 本地临时目录（下载 / 生成 af_lerobot_v3）
WORK_ROOT: Path = Path(tempfile.gettempdir()) / "af_record_migrate_v3"

# 版本
SOURCE_FORMAT_VERSION: str = "v1.0.1"
TARGET_FORMAT_VERSION: str = "v1.0.2"
LEROBOT_V3_DIRNAME: str = "af_lerobot_v3"
AF_META_NAME: str = "af_meta.json"

# 仓库根目录（用于 import process_af_raw / lerobot_v30_writer）
TOOLKIT_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class RecordRow:
    """MySQL 查出的一行；字段名与 MYSQL_QUERY_PENDING 的别名对应。"""

    record_id: Any
    object_prefix: str
    format_version: str
    bucket: str | None = None  # 可选：若 SQL 也选出了 bucket


# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------


def mysql_connect():
    """建立 MySQL 连接。需安装 pymysql。"""
    try:
        import pymysql
    except ImportError as exc:
        raise SystemExit('请安装 pymysql: pip install pymysql') from exc

    if not MYSQL_HOST or not MYSQL_USER or not MYSQL_DATABASE:
        raise SystemExit("请先填写 MYSQL_HOST / MYSQL_USER / MYSQL_DATABASE 等全局配置")

    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset=MYSQL_CHARSET,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def fetch_pending_records(conn) -> list[RecordRow]:
    """执行 MYSQL_QUERY_PENDING，返回待迁移记录列表。"""
    with conn.cursor() as cur:
        cur.execute(MYSQL_QUERY_PENDING)
        rows = cur.fetchall()

    out: list[RecordRow] = []
    for row in rows:
        out.append(
            RecordRow(
                record_id=row["record_id"],
                object_prefix=str(row["object_prefix"]).rstrip("/"),
                format_version=str(row.get("format_version") or ""),
                bucket=row.get("bucket"),
            )
        )
    return out


def update_mysql_format_version(conn, record: RecordRow, new_version: str) -> None:
    """迁移成功后回写 MySQL 中的 format_version。

    TODO: 按真实表名/主键改 SQL。
    """
    sql = """
    UPDATE af_record
    SET format_version = %s, updated_at = NOW()
    WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_version, record.record_id))


# ---------------------------------------------------------------------------
# MinIO
# ---------------------------------------------------------------------------


def minio_client():
    """创建 MinIO 客户端。需安装 minio。"""
    try:
        from minio import Minio
    except ImportError as exc:
        raise SystemExit('请安装 minio: pip install minio') from exc

    if not MINIO_ENDPOINT or not MINIO_ACCESS_KEY or not MINIO_BUCKET:
        raise SystemExit("请先填写 MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_BUCKET 等全局配置")

    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def resolve_bucket(record: RecordRow) -> str:
    return (record.bucket or MINIO_BUCKET).strip()


def list_object_keys(client, bucket: str, prefix: str) -> list[str]:
    """列出 prefix 下全部 object key。"""
    keys: list[str] = []
    for obj in client.list_objects(bucket, prefix=prefix.rstrip("/") + "/", recursive=True):
        if obj.object_name:
            keys.append(obj.object_name)
    return keys


def download_prefix(client, bucket: str, prefix: str, local_dir: Path) -> None:
    """把 MinIO 上 prefix/ 整棵目录树下载到 local_dir。"""
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    prefix = prefix.rstrip("/") + "/"
    for key in list_object_keys(client, bucket, prefix.rstrip("/")):
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix) :]
        if not rel or key.endswith("/"):
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, key, str(dest))


def upload_directory(client, bucket: str, local_dir: Path, remote_prefix: str) -> None:
    """把 local_dir 下文件上传到 remote_prefix/（覆盖同名 key）。"""
    remote_prefix = remote_prefix.rstrip("/")
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{remote_prefix}/{rel}"
        client.fput_object(bucket, key, str(path))


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.stat_object(bucket, key)
        return True
    except Exception:
        return False


def remote_has_lerobot_v3(client, bucket: str, object_prefix: str) -> bool:
    """粗判远端是否已有 af_lerobot_v3（存在 meta/info.json 即视为有）。"""
    marker = f"{object_prefix.rstrip('/')}/{LEROBOT_V3_DIRNAME}/meta/info.json"
    return object_exists(client, bucket, marker)


# ---------------------------------------------------------------------------
# 本地 af-record 处理
# ---------------------------------------------------------------------------


def load_af_meta(record_dir: Path) -> dict[str, Any]:
    path = record_dir / AF_META_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_af_meta(record_dir: Path, meta: dict[str, Any]) -> None:
    path = record_dir / AF_META_NAME
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def bump_format_version(meta: dict[str, Any], target: str = TARGET_FORMAT_VERSION) -> dict[str, Any]:
    meta = dict(meta)
    meta["version"] = target
    return meta


def ensure_toolkit_on_syspath() -> None:
    root = str(TOOLKIT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def generate_af_lerobot_v3(record_dir: Path) -> Path:
    """根据本地 af-record（至少含 af_meta.json + af_rosbag/）生成 af_lerobot_v3/。

    默认走 toolkit 的对齐导出逻辑（与 process_af_raw 一致）。
    若你们希望「只从已有 af_lerobot_v2 转换」，可改写本函数。
    """
    ensure_toolkit_on_syspath()

    from rosbags.highlevel import AnyReader

    from process_af_raw import (
        assert_joint_msgdef,
        bind_cameras,
        collect_aligned_episode,
        export_lerobot_v3_from_episode,
        find_joint_topic,
        load_af_meta,
        parse_cameras,
        parse_joint_names,
        resolve_export_image_keys,
    )

    meta_path = record_dir / AF_META_NAME
    rosbag_dir = record_dir / "af_rosbag"
    if not meta_path.is_file() or not rosbag_dir.is_dir():
        raise FileNotFoundError(
            f"{record_dir} 缺少 af_meta.json 或 af_rosbag/，无法生成 {LEROBOT_V3_DIRNAME}"
        )

    meta = load_af_meta(meta_path)
    cameras = parse_cameras(meta)
    joint_names = parse_joint_names(meta)
    fps = int(round(cameras[0].fps))
    out_dir = record_dir / LEROBOT_V3_DIRNAME

    with AnyReader([rosbag_dir]) as reader:
        bindings = bind_cameras(reader.connections, cameras)
        joint_connection = find_joint_topic(reader.connections)
        assert_joint_msgdef(joint_connection)
        image_key_by_camera = resolve_export_image_keys([b.camera for b in bindings])
        feature_image_keys = [image_key_by_camera[b.camera.name] for b in bindings]
        episode = collect_aligned_episode(
            reader,
            bindings,
            joint_connection,
            joint_names,
            image_key_by_camera,
            task=record_dir.name,
        )
        export_lerobot_v3_from_episode(
            meta,
            episode,
            feature_image_keys,
            out_dir,
            fps,
        )

    return out_dir


def migrate_local_record(record_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """对已下载到本地的一条 af-record 做迁移，返回更新后的 meta。"""
    meta = load_af_meta(record_dir)
    current = str(meta.get("version") or "")
    v3_dir = record_dir / LEROBOT_V3_DIRNAME

    if current not in ("", SOURCE_FORMAT_VERSION, TARGET_FORMAT_VERSION):
        raise RuntimeError(
            f"{record_dir}: unexpected format version {current!r}, "
            f"expected {SOURCE_FORMAT_VERSION!r} or {TARGET_FORMAT_VERSION!r}"
        )

    if v3_dir.is_dir() and not force:
        # 已有 v3：只保证 meta version 升到目标
        if current != TARGET_FORMAT_VERSION:
            meta = bump_format_version(meta)
            save_af_meta(record_dir, meta)
        return meta

    if v3_dir.exists() and force:
        shutil.rmtree(v3_dir)

    generate_af_lerobot_v3(record_dir)
    meta = bump_format_version(load_af_meta(record_dir))
    save_af_meta(record_dir, meta)
    return meta


# ---------------------------------------------------------------------------
# 单条 / 批量迁移（编排）
# ---------------------------------------------------------------------------


def migrate_one(
    client,
    conn,
    record: RecordRow,
    *,
    dry_run: bool = False,
    force: bool = False,
    update_db: bool = True,
) -> None:
    bucket = resolve_bucket(record)
    prefix = record.object_prefix
    local_dir = WORK_ROOT / str(record.record_id)

    print(f"[record {record.record_id}] bucket={bucket} prefix={prefix} version={record.format_version}")

    if record.format_version and record.format_version not in (
        SOURCE_FORMAT_VERSION,
        TARGET_FORMAT_VERSION,
    ):
        print(f"  skip: format_version={record.format_version!r}")
        return

    if remote_has_lerobot_v3(client, bucket, prefix) and not force:
        print(f"  skip: remote already has {LEROBOT_V3_DIRNAME}/ (use --force to rebuild)")
        # 仍可只升 meta version —— 未实现远端只改单个 JSON 的路径时可打开下面分支
        # TODO: 可选实现「仅下载 af_meta.json → 改 version → 回传」
        return

    if dry_run:
        print("  dry-run: would download → generate af_lerobot_v3 → upload → bump version")
        return

    print("  downloading…")
    download_prefix(client, bucket, prefix, local_dir)

    print("  generating af_lerobot_v3 + bumping af_meta version…")
    meta = migrate_local_record(local_dir, force=force)
    print(f"  local meta.version → {meta.get('version')}")

    # 上传：可只上传新增的 af_lerobot_v3/ + af_meta.json，减小流量
    print("  uploading af_meta.json + af_lerobot_v3/…")
    upload_directory(client, bucket, local_dir / LEROBOT_V3_DIRNAME, f"{prefix}/{LEROBOT_V3_DIRNAME}")
    client.fput_object(
        bucket,
        f"{prefix}/{AF_META_NAME}",
        str(local_dir / AF_META_NAME),
    )

    if update_db:
        update_mysql_format_version(conn, record, TARGET_FORMAT_VERSION)
        conn.commit()
        print(f"  mysql format_version → {TARGET_FORMAT_VERSION}")

    # 清理本地临时目录（可选保留便于排错）
    shutil.rmtree(local_dir, ignore_errors=True)
    print("  done")


def iter_batches(items: list[RecordRow], batch_size: int) -> Iterator[list[RecordRow]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run(
    *,
    dry_run: bool = False,
    force: bool = False,
    update_db: bool = True,
    limit: int | None = None,
) -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    client = minio_client()
    conn = mysql_connect()
    try:
        records = fetch_pending_records(conn)
        if limit is not None:
            records = records[:limit]
        print(f"pending records: {len(records)}")
        for record in records:
            try:
                migrate_one(
                    client,
                    conn,
                    record,
                    dry_run=dry_run,
                    force=force,
                    update_db=update_db and not dry_run,
                )
            except Exception as exc:
                conn.rollback()
                print(f"[record {record.record_id}] FAILED: {type(exc).__name__}: {exc}")
                # TODO: 写入失败表 / 重试队列
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 以下为可选扩展点（故意未实现完整逻辑）
# ---------------------------------------------------------------------------


def convert_v2_dir_to_v3(v2_dir: Path, v3_dir: Path) -> None:
    """可选：不读 rosbag，直接把已有 af_lerobot_v2 转成 v3.0 布局。

    官方有 ``convert_dataset_v21_to_v30``；此处若不想引入 lerobot 依赖，
    可自行实现或继续走 ``generate_af_lerobot_v3``（从 rosbag 重导）。
    """
    raise NotImplementedError("可选：从 af_lerobot_v2 转 af_lerobot_v3")


def filter_records(records: Iterable[RecordRow]) -> list[RecordRow]:
    """可选：额外过滤（机器人类型、时间范围、白名单等）。"""
    return list(records)


def notify_done(record: RecordRow) -> None:
    """可选：发消息队列 / 回调业务系统。"""
    return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Migrate MinIO af-record v1.0.1 → add af_lerobot_v3, bump to v1.0.2",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印将要做什么，不下载/上传/改库")
    p.add_argument("--force", action="store_true", help="即使已有 af_lerobot_v3 也重建")
    p.add_argument("--no-update-db", action="store_true", help="不回写 MySQL format_version")
    p.add_argument("--limit", type=int, default=None, help="最多处理 N 条（调试用）")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(
        dry_run=args.dry_run,
        force=args.force,
        update_db=not args.no_update_db,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
