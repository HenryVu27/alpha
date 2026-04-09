"""Message envelope schema for Redis Streams communication."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageEnvelope(BaseModel):
    """Standard message wrapper for all inter-layer communication."""

    timestamp: datetime = Field(default_factory=_utcnow)
    layer: str
    version: int = 1
    payload: dict

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> MessageEnvelope:
        """Deserialize from JSON string."""
        return cls.model_validate_json(data)

    def to_redis(self) -> dict[str, str]:
        """Return dict suitable for Redis XADD."""
        return {"data": self.to_json()}

    @classmethod
    def from_redis(cls, data: dict) -> MessageEnvelope:
        """Reconstruct from Redis XREAD/XRANGE result.

        Handles both str and bytes keys/values as returned by redis-py.
        """
        # Normalise bytes to str
        normalised: dict[str, str] = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            normalised[key] = val
        return cls.from_json(normalised["data"])
