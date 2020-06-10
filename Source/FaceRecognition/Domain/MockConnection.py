from Domain.Connection import Connection
import asyncio


class MockConnection(Connection):
    def __init__(self):
        self.__fake_conn_id = 'fakeid'
        super().__init__(self.__fake_conn_id, None, None)

    async def write_data(self, data):
        await asyncio.sleep(0.1)

    async def read_data(self):
        await asyncio.sleep(0.1)

    @property
    def fake_conn_id(self):
        return self.__fake_conn_id
