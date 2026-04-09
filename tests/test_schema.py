"""Tests for MessageEnvelope schema."""

from datetime import datetime, timezone

import pytest

from common.schema import MessageEnvelope


def test_create_envelope_fields():
    """Creating a MessageEnvelope should populate all fields correctly."""
    env = MessageEnvelope(layer="regime", payload={"state": 2, "prob": 0.85})
    assert env.layer == "regime"
    assert env.version == 1
    assert env.payload == {"state": 2, "prob": 0.85}
    assert isinstance(env.timestamp, datetime)


def test_json_roundtrip():
    """to_json / from_json should produce identical envelope."""
    original = MessageEnvelope(layer="sentiment", payload={"score": -0.3})
    json_str = original.to_json()
    restored = MessageEnvelope.from_json(json_str)
    assert restored.layer == original.layer
    assert restored.version == original.version
    assert restored.payload == original.payload
    assert restored.timestamp == original.timestamp


def test_to_redis_format():
    """to_redis should return dict with 'data' key containing a JSON string."""
    env = MessageEnvelope(layer="regime", payload={"state": 1})
    redis_dict = env.to_redis()
    assert isinstance(redis_dict, dict)
    assert "data" in redis_dict
    assert isinstance(redis_dict["data"], str)


def test_from_redis_roundtrip():
    """from_redis should reconstruct envelope from to_redis output, including bytes."""
    original = MessageEnvelope(layer="correlation", payload={"rho": 0.92})
    redis_dict = original.to_redis()
    restored = MessageEnvelope.from_redis(redis_dict)
    assert restored.layer == original.layer
    assert restored.payload == original.payload

    # Also test with bytes keys/values (as Redis actually returns)
    bytes_dict = {b"data": redis_dict["data"].encode()}
    restored2 = MessageEnvelope.from_redis(bytes_dict)
    assert restored2.layer == original.layer
    assert restored2.payload == original.payload
