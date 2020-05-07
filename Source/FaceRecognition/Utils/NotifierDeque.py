from collections import deque
from typing import Iterable as Iterable
from Utils.Decorators import notify_event, wait_event


class NotifierDeque(deque):

    def __init__(self) -> None:
        super().__init__()
        self._event = None

    @notify_event
    async def extend(self, iterable: Iterable) -> None:
        super().extend(iterable)

    @notify_event
    async def extendleft(self, iterable: Iterable) -> None:
        super().extendleft(iterable)

    @notify_event
    async def append(self, x) -> None:
        super().append(x)

    @notify_event
    async def appendleft(self, x) -> None:
        super().appendleft(x)

    @wait_event(cond=lambda x: len(x) == 0)
    async def pop(self, i: int = ...):
        return super().pop(i)

    @wait_event(cond=lambda x: len(x) == 0)
    async def popleft(self):
        return super().popleft()
