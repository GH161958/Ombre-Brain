from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tools import _runtime as rt
from tools.canonical_link import dispatch


SRC1 = "src_" + "1" * 64
SRC2 = "src_" + "2" * 64


class FakeBucketManager:
    def __init__(self, buckets):
        self.buckets = {bucket["id"]: deepcopy(bucket) for bucket in buckets}
        self.update_calls = []

    async def get(self, bucket_id):
        bucket = self.buckets.get(bucket_id)
        return deepcopy(bucket) if bucket else None

    async def get_including_archive(self, bucket_id):
        return await self.get(bucket_id)

    async def list_all(self, include_archive=False):
        return [deepcopy(bucket) for bucket in self.buckets.values()]

    async def update(self, bucket_id, **kwargs):
        bucket = self.buckets.get(bucket_id)
        if not bucket:
            return False
        self.update_calls.append((bucket_id, deepcopy(kwargs)))
        for key, value in kwargs.items():
            if key == "event_actor":
                continue
            bucket["metadata"][key] = deepcopy(value)
        return True


def make_bucket(bucket_id="b1", title="A real memory"):
    return {
        "id": bucket_id,
        "content": "the remembered event",
        "metadata": {
            "title": title,
            "type": "dynamic",
            "source_links": [
                {"ref": SRC1, "ranges": [[1, 3]], "status": "active"},
                {"ref": SRC2, "ranges": [[4, 5]], "status": "detached"},
            ],
        },
    }


@pytest.fixture
def manager():
    mgr = FakeBucketManager([make_bucket()])
    rt.init(bucket_mgr=mgr, mark_op=None, v3_runtime=None, logger=None)
    return mgr


@pytest.mark.asyncio
async def test_propose_is_read_only_and_returns_evidence_fingerprint(manager):
    result = json.loads(
        await dispatch(
            "propose",
            bucket_id="b1",
            expected_title="A real memory",
            canonical_path="20_VOICE/Aqi_Voice.md",
        )
    )

    assert result["status"] == "review_required"
    assert result["canonical_path"] == "20_VOICE/Aqi_Voice.md"
    assert len(result["evidence"]["evidence_sha256"]) == 64
    assert result["evidence"]["sources"] == [
        {"slot": 1, "ref": SRC1, "ranges": [[1, 3]]}
    ]
    assert manager.update_calls == []


@pytest.mark.asyncio
async def test_confirm_persists_real_vault_receipt_and_read_returns_it(manager):
    proposal = json.loads(
        await dispatch(
            "propose",
            bucket_id="b1",
            expected_title="A real memory",
            canonical_path="20_VOICE/Aqi_Voice.md",
        )
    )
    fingerprint = proposal["evidence"]["evidence_sha256"]

    confirmed = json.loads(
        await dispatch(
            "confirm",
            bucket_id="b1",
            expected_title="A real memory",
            canonical_path="20_VOICE/Aqi_Voice.md",
            expected_evidence_sha256=fingerprint,
            version="0.4.0",
            blob_sha="a" * 40,
            commit_sha="b" * 40,
            history_path="20_VOICE/Voice_History/Aqi_Voice_old.md",
        )
    )

    assert confirmed["status"] == "confirmed"
    assert confirmed["receipt"]["commit_sha"] == "b" * 40
    assert manager.buckets["b1"]["metadata"]["canonical_links"][0]["version"] == "0.4.0"

    read_back = json.loads(
        await dispatch(
            "read",
            bucket_id="b1",
            expected_title="A real memory",
        )
    )
    assert read_back["count"] == 1
    assert read_back["canonical_links"][0]["canonical_path"] == "20_VOICE/Aqi_Voice.md"


@pytest.mark.asyncio
async def test_confirm_rejects_memory_evidence_drift(manager):
    proposal = json.loads(
        await dispatch(
            "propose",
            bucket_id="b1",
            expected_title="A real memory",
            canonical_path="10_IDENTITY/Aqi_Seed.md",
        )
    )
    manager.buckets["b1"]["content"] = "the memory was corrected"

    result = await dispatch(
        "confirm",
        bucket_id="b1",
        expected_title="A real memory",
        canonical_path="10_IDENTITY/Aqi_Seed.md",
        expected_evidence_sha256=proposal["evidence"]["evidence_sha256"],
        version="0.6.3",
        blob_sha="a" * 40,
        commit_sha="b" * 40,
    )

    assert "evidence 已变化" in result
    assert "canonical_links" not in manager.buckets["b1"]["metadata"]


@pytest.mark.asyncio
async def test_confirm_is_idempotent_and_lookup_reverses_link(manager):
    proposal = json.loads(
        await dispatch(
            "propose",
            bucket_id="b1",
            expected_title="A real memory",
            canonical_path="00_HOME/Current.md",
            source_slots=[1],
        )
    )
    kwargs = dict(
        action="confirm",
        bucket_id="b1",
        expected_title="A real memory",
        canonical_path="00_HOME/Current.md",
        source_slots=[1],
        expected_evidence_sha256=proposal["evidence"]["evidence_sha256"],
        version="1",
        blob_sha="a" * 40,
        commit_sha="b" * 40,
    )
    first = json.loads(await dispatch(**kwargs))
    second = json.loads(await dispatch(**kwargs))

    assert first["status"] == "confirmed"
    assert second["status"] == "already_confirmed"
    assert len(manager.buckets["b1"]["metadata"]["canonical_links"]) == 1

    reverse = json.loads(
        await dispatch(
            "lookup",
            canonical_path="00_HOME/Current.md",
        )
    )
    assert reverse["count"] == 1
    assert reverse["matches"][0]["bucket_id"] == "b1"


@pytest.mark.asyncio
async def test_selected_detached_source_is_rejected(manager):
    result = await dispatch(
        "propose",
        bucket_id="b1",
        expected_title="A real memory",
        canonical_path="20_VOICE/Aqi_Voice.md",
        source_slots=[2],
    )
    assert "detached" in result


@pytest.mark.asyncio
async def test_locked_letter_does_not_expose_link_state():
    letter = make_bucket("letter1", "Hidden letter")
    letter["metadata"].update(
        {
            "type": "letter",
            "source_tool": "letter",
            "lock_type": "permanent",
            "unlock_date": "9999-12-31",
            "locked_by": "human",
        }
    )
    mgr = FakeBucketManager([letter])
    rt.init(bucket_mgr=mgr, mark_op=None, v3_runtime=None, logger=None)

    result = await dispatch(
        "read",
        bucket_id="letter1",
        expected_title="Hidden letter",
    )
    assert "尚未" in result
