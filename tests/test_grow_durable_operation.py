import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tools._runtime as rt
from bucket_manager import BucketManager
from errors import PublicToolError
from tools.grow import dispatch
from tools.grow import durable_operation


_RESTART_PROBE = r'''import asyncio
import json
import sys

from bucket_manager import BucketManager
import tools._runtime as rt
from tools.grow import dispatch

class NoopDecay:
    async def ensure_started(self):
        return None

vault, operation_key = sys.argv[1:3]
config = {
    "buckets_dir": vault,
    "matching": {"fuzzy_threshold": 50, "max_results": 5},
    "scoring_weights": {},
    "storage": {"external_change_poll_seconds": 0},
}
rt.bucket_mgr = BucketManager(config, embedding_engine=None)
rt.decay_engine = NoopDecay()
item = {
    "title": "Restart proof",
    "content": "One reviewed candidate remains one exact Memory.",
    "tags": ["acceptance"],
    "domain": ["test"],
    "importance": 5,
}
print(asyncio.run(dispatch(items=[item], operation_key=operation_key, test_data=True)))
'''


class NoopDecay:
    async def ensure_started(self):
        return None


def promotion_item(content="一条已经审阅完成的长期记忆。"):
    return {
        "title": "已审阅事件",
        "content": content,
        "tags": ["关系"],
        "domain": ["生活"],
        "importance": 8,
        "why_remembered": "它对长期连续性有意义。",
    }


@pytest.fixture
def durable_runtime(bucket_mgr, monkeypatch):
    monkeypatch.setattr(rt, "bucket_mgr", bucket_mgr, raising=False)
    monkeypatch.setattr(rt, "decay_engine", NoopDecay(), raising=False)
    return bucket_mgr


async def durable_call(key, item=None):
    return json.loads(
        await dispatch(
            items=[item or promotion_item()],
            operation_key=key,
            test_data=True,
        )
    )


@pytest.mark.asyncio
async def test_first_write_and_retry_return_one_stable_memory(durable_runtime):
    first = await durable_call("aqi-promotion-operation-1")
    second = await durable_call("aqi-promotion-operation-1")

    assert first == {
        "schema": "ombre-memory-write-result/v0",
        "memory_id": first["memory_id"],
        "reused": False,
    }
    assert second == {**first, "reused": True}
    buckets = await durable_runtime.list_all(include_archive=False)
    assert [bucket["id"] for bucket in buckets] == [first["memory_id"]]
    assert buckets[0]["content"] == promotion_item()["content"]


@pytest.mark.asyncio
async def test_concurrent_repeated_calls_still_create_at_most_one_memory(
    durable_runtime
):
    first, second = await asyncio.gather(
        durable_call("aqi-promotion-concurrent"),
        durable_call("aqi-promotion-concurrent"),
    )
    assert first["memory_id"] == second["memory_id"]
    assert {first["reused"], second["reused"]} == {False, True}
    assert len(await durable_runtime.list_all(include_archive=False)) == 1


@pytest.mark.asyncio
async def test_lost_response_recovers_after_bucket_manager_restart(
    durable_runtime, test_config, fake_embedding_engine, monkeypatch
):
    lost = await durable_call("aqi-promotion-restart")
    assert lost["reused"] is False

    restarted = BucketManager(test_config, embedding_engine=fake_embedding_engine)
    monkeypatch.setattr(rt, "bucket_mgr", restarted, raising=False)
    recovered = await durable_call("aqi-promotion-restart")

    assert recovered == {**lost, "reused": True}
    buckets = await restarted.list_all(include_archive=False)
    assert len(buckets) == 1
    assert buckets[0]["id"] == lost["memory_id"]


@pytest.mark.asyncio
async def test_same_operation_key_with_conflicting_payload_fails_without_write(
    durable_runtime
):
    first = await durable_call("aqi-promotion-conflict")
    with pytest.raises(PublicToolError) as caught:
        await durable_call(
            "aqi-promotion-conflict",
            promotion_item("不同的候选正文不得复用同一 operation key。"),
        )

    assert "不同的写入请求" in caught.value.public_message
    buckets = await durable_runtime.list_all(include_archive=False)
    assert len(buckets) == 1
    assert buckets[0]["id"] == first["memory_id"]


@pytest.mark.asyncio
async def test_failure_before_markdown_creation_leaves_truthfully_retryable_claim(
    durable_runtime, monkeypatch
):
    original_create = durable_runtime.create

    async def fail_before_create(**_kwargs):
        raise OSError("injected pre-write failure")

    monkeypatch.setattr(durable_runtime, "create", fail_before_create)
    with pytest.raises(OSError, match="injected pre-write failure"):
        await durable_call("aqi-promotion-pre-write-failure")
    assert await durable_runtime.list_all(include_archive=False) == []

    monkeypatch.setattr(durable_runtime, "create", original_create)
    retried = await durable_call("aqi-promotion-pre-write-failure")
    assert retried["reused"] is False
    assert len(await durable_runtime.list_all(include_archive=False)) == 1


@pytest.mark.asyncio
async def test_write_before_receipt_commit_is_recovered_without_second_memory(
    durable_runtime, monkeypatch
):
    original_succeed = durable_operation.DurableGrowOperationStore.succeed
    failed_once = False

    def lose_receipt_once(store, record):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("injected receipt commit loss")
        return original_succeed(store, record)

    monkeypatch.setattr(
        durable_operation.DurableGrowOperationStore,
        "succeed",
        lose_receipt_once,
    )
    with pytest.raises(OSError, match="receipt commit loss"):
        await durable_call("aqi-promotion-receipt-loss")
    buckets_after_loss = await durable_runtime.list_all(include_archive=False)
    assert len(buckets_after_loss) == 1

    recovered = await durable_call("aqi-promotion-receipt-loss")
    assert recovered["reused"] is True
    buckets_after_retry = await durable_runtime.list_all(include_archive=False)
    assert [bucket["id"] for bucket in buckets_after_retry] == [
        recovered["memory_id"]
    ]


@pytest.mark.asyncio
async def test_operation_path_is_one_item_only_and_does_not_store_raw_key(
    durable_runtime
):
    with pytest.raises(PublicToolError):
        await dispatch(
            items=[promotion_item(), promotion_item("第二条")],
            operation_key="must-not-split",
            test_data=True,
        )

    raw_key = "opaque-private-operation-key"
    result = await durable_call(raw_key)
    bucket = await durable_runtime.get(result["memory_id"])
    operation = bucket["metadata"]["write_operation"]
    assert operation["schema"] == "ombre-write-operation/v0"
    assert len(operation["key_sha256"]) == 64
    assert len(operation["request_sha256"]) == 64
    assert raw_key not in Path(bucket["path"]).read_text(encoding="utf-8")
    assert "source_refs" not in bucket["metadata"]


def test_durable_result_is_recovered_by_a_fresh_process(tmp_path):
    vault = tmp_path / "restart-vault"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "OMBRE_VAULT_DIR": str(vault),
        "OMBRE_BUCKETS_DIR": str(vault),
    }
    command = [
        sys.executable, "-c", _RESTART_PROBE,
        str(vault), "aqi-real-process-restart",
    ]
    first = subprocess.run(
        command, check=True, capture_output=True, text=True, env=environment
    )
    second = subprocess.run(
        command, check=True, capture_output=True, text=True, env=environment
    )
    first_result = json.loads(first.stdout.strip())
    second_result = json.loads(second.stdout.strip())

    assert first_result["reused"] is False
    assert second_result == {**first_result, "reused": True}
    memory_files = [
        path for path in vault.rglob("*.md")
        if path.parts[-2] not in {"_ledger", "_media"}
    ]
    assert len(memory_files) == 1
