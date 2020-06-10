from Domain.Connection import Connection
from Domain.MockConnection import MockConnection
from Utils.StringUtils import get_random_alphanumeric_string


class ConnectionManager:
    def __init__(self, conn_id_len: int = 10):
        self.__conn_id_len = conn_id_len
        self.__fake_conn = MockConnection()
        self.__connections = {}

    def register_connection(self, reader_stream, writer_stream) -> Connection:
        conn_id = get_random_alphanumeric_string(self.__conn_id_len)
        conn = Connection(conn_id, reader_stream, writer_stream)
        self.__connections[conn_id] = conn
        return conn

    def unregister_connection(self, conn_id, close_conn: bool = True) -> None:
        try:
            if close_conn:
                self.__connections[conn_id].close()
            del self.__connections[conn_id]
        except KeyError:
            raise ConnectionError(f"No connection with id {conn_id}")

    def get_connection(self, conn_id) -> Connection:
        try:
            conn = self.__fake_conn if conn_id == self.__fake_conn.fake_conn_id else self.__connections[conn_id]
        except KeyError:
            raise ConnectionError(f"No connection with id {conn_id}")
        return conn
