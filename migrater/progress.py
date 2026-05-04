import threading
from collections import deque
from typing import Any


class ProgressBuffer:
    """
    Thread-safe bounded buffer for transfer progress events.

    If producers (migrater) are faster than consumers (token bucket),
    old events are dropped instead of allowing unbounded memory growth.
    """

    def __init__(self, maxlen: int = 128):
        """
        Initialize the progress buffer.

        Args:
            maxlen(int):
                The maximum number of progress events to store in the buffer.
                If the buffer is full, the oldest event will be dropped.
                Default is 128.
        """
        self._lock = threading.Lock()
        self._items = deque(maxlen=maxlen)

    def add(self, message: dict[str, Any]) -> None:
        """
        Add a progress event to the buffer.

        If the buffer is full, the oldest event will be dropped.

        Args:
            message(dict[str, Any]): The progress event to add.

        Example:
            progress_buffer.add({
                "timestamp": 1714824000.123,
                "bytes_copied": 10485760,
                "total_bytes": 52428800,
                "remaining_bytes": 41943040,
                "interval_seconds": 0.5,
                "interval_throughput_bps": 1000000.0,
                "average_throughput_bps": 995000.0,
                "configured_rate_bps": 1000000.0,
                "chunk_size_bytes": 65536,
            })
        """
        with self._lock:
            self._items.append(message)

    def pop(self) -> list[dict[str, Any]]:
        """
        Pop all progress events from the buffer.

        Returns:
            list[dict[str, Any]]: A list of progress events.
        """
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items
