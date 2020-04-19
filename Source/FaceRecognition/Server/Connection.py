import pickle


class Connection:
    def __init__(self, conn_id, reader_stream, writer_stream, byte_order='big'):
        self.__conn_id = conn_id
        self.__reader_stream = reader_stream
        self.__writer_stream = writer_stream
        self.__byte_order = byte_order

    @property
    def connection_id(self):
        return self.__conn_id

    @property
    def reader_stream(self):
        return self.__reader_stream

    @property
    def writer_stream(self):
        return self.__writer_stream

    async def read_data(self):
        data_len = int.from_bytes(await self.__reader_stream.read(4), byteorder=self.__byte_order)
        if data_len == 0:
            data = None
        else:
            data = await self.__reader_stream.read(data_len)
        return data

    async def write_data(self, data):
        self.__writer_stream.write(len(data).to_bytes(4, self.__byte_order))
        self.__writer_stream.write(data)
        await self.__writer_stream.drain()

    def close(self):
        self.__writer_stream.close()
