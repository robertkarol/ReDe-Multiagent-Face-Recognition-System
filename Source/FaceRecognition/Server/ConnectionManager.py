import random
import string

from Server.Connection import Connection


class ConnectionManager:
    def __init__(self, conn_id_len=10):
        self.__conn_id_len = conn_id_len
        self.__connections = {}

    def register_connection(self, reader_stream, writer_stream):
        conn_id = self.__get_random_alphanumeric_string(self.__conn_id_len)
        conn = Connection(conn_id, reader_stream, writer_stream)
        self.__connections[conn_id] = conn
        return conn

    def unregister_connection(self, conn_id, close_conn=True):
        if close_conn:
            self.__connections[conn_id].close()
        del self.__connections[conn_id]

    def get_connection(self, conn_id):
        return self.__connections[conn_id]

    def __get_random_alphanumeric_string(self, string_length):
        charset = string.ascii_letters + string.digits
        return ''.join((random.choice(charset) for _ in range(string_length)))
