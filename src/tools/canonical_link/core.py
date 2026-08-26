"""Memory ↔ Vault canonical link handshake.

`propose` is read-only: it validates a precise Memory bucket and returns a
stable evidence fingerprint for review. `confirm` may be called only after a
real Vault mutation succeeds; it persists that Vault receipt on the original
Memory bucket. `read` and `lookup` expose the durable cross-reference.

This module never calls Vault and never promotes Memory by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from ombrebrain.storage.source_store import source_links_from_metadata
from utils import normalize_memory_title

from .. import _runtime as rt
from ..plan.core import is_letter_bucket, letter_lock_state

_MAX_LINKS_PER_BUCKET = 32
_MAX_LOOKUP_RESULTS = 50
_MAX_PATH_CHARS = 500
_MAX_VERSION_CHARS = 80
_MAX_NOTE_CHARS = 500
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _clean_title(value: object) -> str:
    try:
        return normalize_memory_title(value)
    except ValueError:
        return " ".join(str(value or "").split())


def _safe_path(value: object, *, field: str = "canonical_path") -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw or len(raw) > _MAX_PATH_CHARS:
        raise ValueError(f"{field} 为空或过长。")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} 必须是安全的 Vault 相对路径。")
    normalized = path.as_posix()
    if not normalized.endswith(".md"):
        raise ValueError(f"{field} 必须指向 Markdown canonical 文件。")
    return normalized


def _safe_sha(value: object, *, field: str) -> str:
    raw = str(value or "").strip().lower()
    if not _SHA_RE.fullmatch(raw):
        raise ValueError(f"{field} 必须是 7-64 位十六进制 SHA。")
    return raw


def _safe_version(value: object) -> str:
    raw = str(value or "").strip()
    if not raw or len(raw) > _MAX_VERSION_CHARS:
        raise ValueError("version 不能为空或过长。")
    return raw


def _normalize_source_slots(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("source_slots 必须是整数列表。")
    slots: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError("source_slots 必须是正整数列表。")
        if item not in slots:
            slots.append(item)
    return sorted(slots)


def _normalize_receipts(value: object) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        return []
    receipts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("canonical_path") or "").strip()
        commit_sha = str(item.get("commit_sha") or "").strip().lower()
        if not path or not commit_sha:
            continue
        receipts.append(dict(item))
    return receipts[:_MAX_LINKS_PER_BUCKET]


async def _load_precise_bucket(
    bucket_id: object,
    expected_title: object,
) -> tuple[dict[str, Any] | None, str]:
    normalized_id = str(bucket_id or "").strip()
    title = _clean_title(expected_title)
    if not normalized_id or not title:
        return None, "canonical_link 需要 bucket_id 和 expected_title。"

    getter = getattr(rt.bucket_mgr, "get_including_archive", None)
    if not callable(getter):
        getter = rt.bucket_mgr.get
    try:
        bucket = await getter(normalized_id)
    except Exception:
        return None, "读取 Memory bucket 失败。"
    if not bucket:
        return None, f"未找到 Memory bucket: {normalized_id}"

    metadata = bucket.get("metadata") or {}
    actual_title = _clean_title(metadata.get("title"))
    if not actual_title:
        return None, "该 Memory bucket 没有显式 title，拒绝建立 canonical link。"
    if actual_title != title:
        return None, "标题不匹配，拒绝建立 canonical link。"

    if is_letter_bucket(bucket) and letter_lock_state(bucket, "ai")["locked"]:
        return None, "这封 Letter 尚未向当前一方开放，不能读取或建立 canonical link。"
    return bucket, ""


def _evidence_snapshot(
    bucket: dict[str, Any],
    requested_slots: list[int] | None,
) -> dict[str, Any]:
    metadata = bucket.get("metadata") or {}
    try:
        links = source_links_from_metadata(metadata)
    except ValueError as exc:
        raise ValueError(f"Memory Source metadata 无效：{exc}") from exc

    if requested_slots is None:
        selected_slots = [
            index
            for index, link in enumerate(links, 1)
            if link.get("status") == "active"
        ]
    else:
        selected_slots = requested_slots

    sources: list[dict[str, Any]] = []
    for slot in selected_slots:
        if slot > len(links):
            raise ValueError(f"source_slot={slot} 不存在。")
        link = links[slot - 1]
        if link.get("status") != "active":
            raise ValueError(f"source_slot={slot} 已 detached，不能作为本次晋升证据。")
        sources.append(
            {
                "slot": slot,
                "ref": link["ref"],
                "ranges": link.get("ranges") or [],
            }
        )

    title = _clean_title(metadata.get("title"))
    body = str(bucket.get("content") or "")
    payload = {
        "bucket_id": str(bucket.get("id") or ""),
        "title": title,
        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "sources": sources,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


async def dispatch(
    action: str,
    bucket_id: str = "",
    expected_title: str = "",
    canonical_path: str = "",
    source_slots: list[int] | None = None,
    expected_evidence_sha256: str = "",
    version: str = "",
    blob_sha: str = "",
    commit_sha: str = "",
    history_path: str = "",
    note: str = "",
    limit: int = 20,
) -> str:
    action = str(action or "").strip().lower()
    if action not in {"propose", "confirm", "read", "lookup"}:
        return "action 仅支持 propose / confirm / read / lookup。"

    if rt.mark_op:
        rt.mark_op("canonical_link")
    rt.record_v3_tool_event(
        "canonical_link",
        {
            "action": action,
            "bucket_id": str(bucket_id or ""),
            "canonical_path_length": len(str(canonical_path or "")),
            "source_slots_count": len(source_slots or []) if isinstance(source_slots, list) else 0,
        },
    )

    if action == "lookup":
        try:
            path = _safe_path(canonical_path)
            normalized_limit = max(1, min(_MAX_LOOKUP_RESULTS, int(limit)))
        except (TypeError, ValueError, OverflowError) as exc:
            return str(exc)

        try:
            all_buckets = await rt.bucket_mgr.list_all(include_archive=True)
        except Exception:
            return "读取 Memory canonical link index 失败。"

        matches: list[dict[str, Any]] = []
        for bucket in all_buckets:
            if is_letter_bucket(bucket) and letter_lock_state(bucket, "ai")["locked"]:
                continue
            metadata = bucket.get("metadata") or {}
            for receipt in _normalize_receipts(metadata.get("canonical_links")):
                if receipt.get("canonical_path") != path:
                    continue
                matches.append(
                    {
                        "bucket_id": bucket.get("id", ""),
                        "title": _clean_title(metadata.get("title")),
                        "type": metadata.get("type", ""),
                        "receipt": receipt,
                    }
                )
                if len(matches) >= normalized_limit:
                    break
            if len(matches) >= normalized_limit:
                break
        return _render(
            {
                "action": "lookup",
                "canonical_path": path,
                "count": len(matches),
                "matches": matches,
            }
        )

    bucket, error = await _load_precise_bucket(bucket_id, expected_title)
    if error:
        return error
    assert bucket is not None

    metadata = bucket.get("metadata") or {}
    if action == "read":
        receipts = _normalize_receipts(metadata.get("canonical_links"))
        return _render(
            {
                "action": "read",
                "bucket_id": bucket.get("id", ""),
                "title": _clean_title(metadata.get("title")),
                "count": len(receipts),
                "canonical_links": receipts,
            }
        )

    try:
        path = _safe_path(canonical_path)
        slots = _normalize_source_slots(source_slots)
        evidence = _evidence_snapshot(bucket, slots)
        clean_note = str(note or "").strip()[:_MAX_NOTE_CHARS]
    except ValueError as exc:
        return str(exc)

    if action == "propose":
        return _render(
            {
                "action": "propose",
                "status": "review_required",
                "canonical_path": path,
                "note": clean_note,
                "evidence": evidence,
                "next": (
                    "EE × Aqi review → fresh Vault read/SHA → approved Vault update → "
                    "canonical_link(action='confirm', expected_evidence_sha256=...)"
                ),
            }
        )

    # confirm
    expected = str(expected_evidence_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return "confirm 需要 propose 返回的 64 位 expected_evidence_sha256。"
    if evidence["evidence_sha256"] != expected:
        return (
            "Memory evidence 已变化，拒绝写入 canonical receipt。"
            "请重新 propose / review 后再确认。"
        )

    try:
        clean_version = _safe_version(version)
        clean_blob_sha = _safe_sha(blob_sha, field="blob_sha")
        clean_commit_sha = _safe_sha(commit_sha, field="commit_sha")
        clean_history = (
            _safe_path(history_path, field="history_path")
            if str(history_path or "").strip()
            else ""
        )
    except ValueError as exc:
        return str(exc)

    receipts = _normalize_receipts(metadata.get("canonical_links"))
    for receipt in receipts:
        if (
            receipt.get("canonical_path") == path
            and str(receipt.get("commit_sha") or "").lower() == clean_commit_sha
        ):
            return _render(
                {
                    "action": "confirm",
                    "status": "already_confirmed",
                    "bucket_id": bucket.get("id", ""),
                    "receipt": receipt,
                }
            )

    if len(receipts) >= _MAX_LINKS_PER_BUCKET:
        return (
            f"该 Memory bucket 已有 {_MAX_LINKS_PER_BUCKET} 条 canonical receipts；"
            "拒绝静默丢弃旧记录。"
        )

    receipt = {
        "canonical_path": path,
        "version": clean_version,
        "blob_sha": clean_blob_sha,
        "commit_sha": clean_commit_sha,
        "history_path": clean_history,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence,
    }
    if clean_note:
        receipt["note"] = clean_note

    try:
        committed = await rt.bucket_mgr.update(
            str(bucket.get("id") or ""),
            canonical_links=[*receipts, receipt],
            event_actor="canonical_link",
        )
    except Exception:
        return "canonical receipt 写回 Memory 失败。"
    if not committed:
        return "canonical receipt 写回 Memory 被拒绝；未修改任何 canonical。"

    return _render(
        {
            "action": "confirm",
            "status": "confirmed",
            "bucket_id": bucket.get("id", ""),
            "receipt": receipt,
        }
    )
