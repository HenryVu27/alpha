"""Tests for RedisStreamClient (no live Redis needed)."""

import pytest

from common.redis_client import RedisStreamClient


def test_client_init_stores_maxlen():
    """Client init should store stream_maxlen."""
    client = RedisStreamClient(host="localhost", port=6379, db=0, stream_maxlen=5000)
    assert client.stream_maxlen == 5000


def test_stream_key():
    """stream_key should return 'stream:{layer}'."""
    client = RedisStreamClient(host="localhost", port=6379, db=0, stream_maxlen=10000)
    assert client.stream_key("regime") == "stream:regime"
    assert client.stream_key("sentiment") == "stream:sentiment"
