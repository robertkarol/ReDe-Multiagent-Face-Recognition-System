import asyncio

from Server.Connection import Connection


class MockConnection(Connection):
    def __init__(self):
        super().__init__('abc', None, None)

    async def write_data(self, data):
        await asyncio.sleep(0.1)

    async def read_data(self):
        await asyncio.sleep(0.1)
