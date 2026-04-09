"""Async Redis Streams client for inter-layer communication."""

from __future__ import annotations

from typing import Optional

import redis.asyncio as aioredis

from common.schema import MessageEnvelope


class RedisStreamClient:
    """Thin async wrapper around Redis Streams for publish/subscribe."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        stream_maxlen: int = 10_000,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.stream_maxlen = stream_maxlen
        self._redis: Optional[aioredis.Redis] = None

    @staticmethod
    def stream_key(layer: str) -> str:
        """Return the Redis stream key for a given layer."""
        return f"stream:{layer}"

    async def connect(self) -> None:
        """Create the async Redis connection."""
        self._redis = aioredis.Redis(
            host=self.host, port=self.port, db=self.db
        )

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def publish(self, layer: str, payload: dict) -> str:
        """Wrap payload in a MessageEnvelope and publish to the layer's stream.

        Returns the stream entry ID.
        """
        assert self._redis is not None, "Call connect() first"
        envelope = MessageEnvelope(layer=layer, payload=payload)
        entry_id: bytes = await self._redis.xadd(
            self.stream_key(layer),
            envelope.to_redis(),
            maxlen=self.stream_maxlen,
        )
        return entry_id.decode() if isinstance(entry_id, bytes) else entry_id

    async def subscribe(
        self,
        layers: list[str],
        last_ids: Optional[dict[str, str]] = None,
        block_ms: int = 5000,
    ) -> list[tuple[str, MessageEnvelope]]:
        """Read new messages from one or more layer streams via XREAD.

        Returns a list of (stream_key, MessageEnvelope) tuples.
        """
        assert self._redis is not None, "Call connect() first"
        if last_ids is None:
            last_ids = {}
        streams = {
            self.stream_key(layer): last_ids.get(layer, "$")
            for layer in layers
        }
        result = await self._redis.xread(streams, block=block_ms)
        if result is None:
            return []
        messages: list[tuple[str, MessageEnvelope]] = []
        for stream_name, entries in result:
            key = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
            for _entry_id, data in entries:
                envelope = MessageEnvelope.from_redis(data)
                messages.append((key, envelope))
        return messages

    async def get_latest(self, layer: str) -> Optional[MessageEnvelope]:
        """Return the most recent message from a layer's stream, or None."""
        assert self._redis is not None, "Call connect() first"
        result = await self._redis.xrevrange(
            self.stream_key(layer), count=1
        )
        if not result:
            return None
        _entry_id, data = result[0]
        return MessageEnvelope.from_redis(data)
