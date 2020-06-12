import asyncio


def notify_event(func):
    async def wrapper(*args, **kwargs):
        self = args[0]
        if not self._event:
            self._event = asyncio.Condition()
        res = await func(*args, **kwargs)
        async with self._event:
            self._event.notify_all()
        return res
    return wrapper


def wait_event(cond):
    def wait_event_real(func):
        async def wrapper(*args, **kwargs):
            self = args[0]
            if not self._event:
                self._event = asyncio.Condition()
            if cond(self):
                async with self._event:
                    await self._event.wait()
            return await func(*args, **kwargs)
        return wrapper
    return wait_event_real
