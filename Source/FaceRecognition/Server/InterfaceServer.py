from Server.ConnectionManager import ConnectionManager
from concurrent import futures
import asyncio
import multiprocessing
import queue


class InterfaceServer(multiprocessing.Process):
    def __init__(self, requests: multiprocessing.Queue, responses: multiprocessing.Queue, ip, port, max_threads_count=4):
        super().__init__()
        self.requests = requests
        self.responses = responses
        self.__ip = ip
        self.__port = port
        self.__max_threads_count = max_threads_count
        self.__loop = None
        self.__connection_manager = ConnectionManager()

    def enqueue_responses(self, responses):
        for res in responses:
            self.responses.put(res)

    def dequeue_requests(self, amount=-1):
        req = []
        if amount == -1:
            amount = self.requests.qsize()
        if amount > 0:
            req = [self.requests.get()]
            try:
                for _ in range(amount - 1):
                    req.append(self.requests.get_nowait())
            except queue.Empty:
                print(amount)
                print("error")
                pass
        return req

    def run(self):
        executor = futures.ThreadPoolExecutor(max_workers=self.__max_threads_count)
        self.__loop = asyncio.get_event_loop()
        self.__loop.set_default_executor(executor)
        try:
            asyncio.ensure_future(self.__responses_handler())
            asyncio.ensure_future(self.__start_requests_server())
            self.__loop.run_forever()
        except Exception:
            pass
        finally:
            self.__loop.close()

    async def __start_requests_server(self):
        req_server = await asyncio.start_server(
            self.__requests_handler, self.__ip, self.__port)
        async with req_server:
            await req_server.serve_forever()

    async def __requests_handler(self, reader, writer):
        current_conn = self.__connection_manager.register_connection(reader, writer)
        while True:
            print("Processing requests...")
            data = await current_conn.read_data()
            if not data:
                break
            self.requests.put((current_conn.connection_id, data))
        self.__connection_manager.unregister_connection(current_conn.connection_id)
        print("Ending processing requests...")

    async def __responses_handler(self):
        while True:
            print("Processing responses...")
            current_conn, message = await self.__loop.run_in_executor(None, lambda: self.responses.get())
            await self.__connection_manager.get_connection(current_conn).write_data(message)
            print(f"Sending: {message}")
