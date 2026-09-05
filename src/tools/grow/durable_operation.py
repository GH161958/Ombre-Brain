"""Durable, exactly-once recovery for one pre-formed ``grow`` item.

This path is opt-in. Ordinary ``grow`` calls continue to use their existing
split/merge behavior. The durable operation record and the deterministic,
no-overwrite Markdown ID form a recoverable pair across process restarts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from errors import PublicToolError

from .. import _runtime as rt


_SCHEMA_VERSION = 1
_RESULT_SCHEMA = "ombre-memory-write-result/v0"
_OPERATION_SCHEMA = "ombre-write-operation/v0"
_ALLOWED_ITEM_FIELDS = {
    "content", "title", "tags", "domain", "importance", "why_remembered"
}
_MAX_OPERATION_KEY_BYTES = 512


@dataclass(frozen=True)
class OperationRecord:
    key_hash: str
    request_hash: str
    memory_id: str
    status: str


class OperationConflict(RuntimeError):
    pass


class OperationStateError(RuntimeError):
    pass


class DurableGrowOperationStore:
    """Small SQLite authority for operation-key to Memory-ID receipts."""

    def __init__(self, vault_dir: str):
        ledger_dir = Path(vault_dir) / "_ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        self.path = ledger_dir / "grow_operations.sqlite3"
        self.connection = sqlite3.connect(str(self.path), timeout=5)
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA busy_timeout=5000")
            self._initialize()
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _SCHEMA_VERSION:
            raise OperationStateError("unsupported durable grow operation schema")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS grow_operations (
                    key_hash TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    memory_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('prepared', 'succeeded')),
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info(grow_operations)"
                ).fetchall()
            }
            required = {
                "key_hash", "request_hash", "memory_id", "status",
                "created_at", "updated_at"
            }
            if not required.issubset(columns):
                raise OperationStateError("invalid durable grow operation schema")
            self.connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @staticmethod
    def _record(row: tuple[Any, ...] | None) -> OperationRecord | None:
        return OperationRecord(*row) if row else None

    def get(self, key_hash: str) -> OperationRecord | None:
        return self._record(
            self.connection.execute(
                """
                SELECT key_hash, request_hash, memory_id, status
                FROM grow_operations WHERE key_hash = ?
                """,
                (key_hash,),
            ).fetchone()
        )

    def claim(self, key_hash: str, request_hash: str, memory_id: str) -> OperationRecord:
        now = time.time_ns() // 1_000_000
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.get(key_hash)
            if existing:
                if existing.request_hash != request_hash:
                    raise OperationConflict("operation key payload conflict")
                self.connection.commit()
                return existing
            self.connection.execute(
                """
                INSERT INTO grow_operations (
                    key_hash, request_hash, memory_id, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'prepared', ?, ?)
                """,
                (key_hash, request_hash, memory_id, now, now),
            )
            self.connection.commit()
            return OperationRecord(key_hash, request_hash, memory_id, "prepared")
        except Exception:
            self.connection.rollback()
            raise

    def succeed(self, record: OperationRecord) -> OperationRecord:
        now = time.time_ns() // 1_000_000
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            changed = self.connection.execute(
                """
                UPDATE grow_operations SET status = 'succeeded', updated_at = ?
                WHERE key_hash = ? AND request_hash = ? AND memory_id = ?
                  AND status IN ('prepared', 'succeeded')
                """,
                (now, record.key_hash, record.request_hash, record.memory_id),
            ).rowcount
            if changed != 1:
                raise OperationStateError("durable grow operation changed concurrently")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return OperationRecord(
            record.key_hash, record.request_hash, record.memory_id, "succeeded"
        )


def _operation_hash(operation_key: str) -> str:
    if not isinstance(operation_key, str):
        raise PublicToolError("operation_key 必须是非空字符串，未创建任何桶。")
    encoded = operation_key.encode("utf-8")
    if (
        not operation_key
        or not operation_key.strip()
        or len(encoded) > _MAX_OPERATION_KEY_BYTES
        or any(ord(char) < 32 for char in operation_key)
    ):
        raise PublicToolError("operation_key 不合法，未创建任何桶。")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_item(item: dict[str, Any], test_data: bool) -> tuple[dict[str, Any], str]:
    unknown = sorted(set(item) - _ALLOWED_ITEM_FIELDS)
    if unknown:
        raise PublicToolError("幂等单条写入包含未支持字段，未创建任何桶。")
    content = item.get("content")
    title = item.get("title")
    if not isinstance(content, str) or not content.strip():
        raise PublicToolError("幂等单条写入需要非空 content，未创建任何桶。")
    if not isinstance(title, str) or not title.strip():
        raise PublicToolError("幂等单条写入需要明确 title，未创建任何桶。")

    normalized = {
        "content": content.strip(),
        "title": title.strip(),
        "tags": item.get("tags") or [],
        "domain": item.get("domain") or ["未分类"],
        "importance": item.get("importance") if item.get("importance") is not None else 5,
        "why_remembered": str(item.get("why_remembered") or "").strip(),
        "test_data": bool(test_data),
    }
    canonical = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return normalized, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _matches_operation(bucket: dict[str, Any] | None, record: OperationRecord) -> bool:
    metadata = bucket.get("metadata") if isinstance(bucket, dict) else None
    operation = metadata.get("write_operation") if isinstance(metadata, dict) else None
    return isinstance(operation, dict) and operation == {
        "schema": _OPERATION_SCHEMA,
        "key_sha256": record.key_hash,
        "request_sha256": record.request_hash,
    }


def _result(memory_id: str, reused: bool) -> str:
    return json.dumps(
        {
            "schema": _RESULT_SCHEMA,
            "memory_id": memory_id,
            "reused": bool(reused),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


async def execute_durable_item(
    item: dict[str, Any], operation_key: str, *, test_data: bool = False
) -> str:
    """Create or recover exactly one pre-formed Memory for one operation key."""

    key_hash = _operation_hash(operation_key)
    normalized, request_hash = _canonical_item(item, test_data)
    memory_id = f"op_{key_hash}"
    store = DurableGrowOperationStore(rt.bucket_mgr.base_dir)
    try:
        try:
            record = store.claim(key_hash, request_hash, memory_id)
        except OperationConflict as exc:
            raise PublicToolError(
                "operation_key 已用于不同的写入请求，未创建任何桶。"
            ) from exc

        bucket = await rt.bucket_mgr.get(record.memory_id)
        if bucket is not None:
            if not _matches_operation(bucket, record):
                raise PublicToolError(
                    "operation_key 对应的持久化记录冲突，未创建任何桶。"
                )
            store.succeed(record)
            return _result(record.memory_id, reused=True)
        if record.status == "succeeded":
            raise PublicToolError(
                "operation_key 的原始 Memory 已不可用，拒绝重复创建。"
            )

        try:
            await rt.bucket_mgr.create(
                content=normalized["content"],
                title=normalized["title"],
                tags=normalized["tags"],
                domain=normalized["domain"],
                importance=normalized["importance"],
                why_remembered=normalized["why_remembered"],
                source_tool="grow",
                test_data=normalized["test_data"],
                bucket_id_override=record.memory_id,
                require_exact_bucket_id=True,
                write_operation={
                    "key_sha256": record.key_hash,
                    "request_sha256": record.request_hash,
                },
            )
        except Exception:
            # Markdown is committed before derived-index work. Recover a write
            # that completed even if a later step or response path failed.
            bucket = await rt.bucket_mgr.get(record.memory_id)
            if not _matches_operation(bucket, record):
                raise

        store.succeed(record)
        return _result(record.memory_id, reused=False)
    finally:
        store.close()
