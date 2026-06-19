"""事件总线:基于 heapq 的优先级队列。

按 ``(timestamp, priority, seq)`` 排序,保证同时间戳下事件按优先级处理,
同优先级按入队顺序(FIFO)。
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator

from djinn.engine.events import Event


class EventBus:
    """优先级事件队列。"""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = 0

    def push(self, event: Event) -> None:
        if event.seq == 0:
            self._counter += 1
            event.seq = self._counter
        heapq.heappush(self._heap, event)

    def pop(self) -> Event | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Event | None:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def drain(self) -> Iterator[Event]:
        """排空总线,按序产出所有事件。"""
        while self._heap:
            yield heapq.heappop(self._heap)
