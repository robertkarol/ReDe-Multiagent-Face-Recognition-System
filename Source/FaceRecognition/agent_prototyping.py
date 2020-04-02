# DISCLAIMER: This code is a bit ugly
import multiprocessing
import queue
import time
from concurrent import futures
import asyncio
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from DatasetHelpers import DatasetHelpers
from RecognitionModel import RecognitionModel

start_time = None  # start_time - time when the first agent is done loading a model and starts resolving
end_time = None  # end_time - time when the last running agent is done


class Blackboard:
    '''
    Mock blackboard. Suppose we have uniform amount of "requests" for each agent
    '''
    agent1 = []
    agent2 = []
    agent3 = []
    agent4 = []
    agent5 = []
    results = []
    # TODO: use deque instead of lists for the real blackboard

    def __init__(self):
        images_to_predict = []
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/robi'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/mindy_kaling'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/madonna'))
        # images_to_predict is 16; we make it 320 for each agent => 1600 total
        self.agent1 = images_to_predict[:] * 20
        self.agent2 = images_to_predict[:] * 20
        self.agent3 = images_to_predict[:] * 20
        self.agent4 = images_to_predict[:] * 20
        self.agent5 = images_to_predict[:] * 20

    def poll(self, agent, amount):
        if agent == 1:
            dt = self.agent1[-1 - amount:-1]
            del self.agent1[-1 - amount:-1]
            return dt
        if agent == 2:
            dt = self.agent2[-1 - amount:-1]
            del self.agent2[-1 - amount:-1]
            return dt
        if agent == 3:
            dt = self.agent3[-1 - amount:-1]
            del self.agent3[-1 - amount:-1]
            return dt
        if agent == 4:
            dt = self.agent4[-1 - amount:-1]
            del self.agent4[-1 - amount:-1]
            return dt
        if agent == 5:
            dt = self.agent5[-1 - amount:-1]
            del self.agent5[-1 - amount:-1]
            return dt

    def poll_results(self, amount=-1):
        if amount == -1:
            dt = self.results
            self.results = []
            return dt
        dt = self.results[-1 - amount:-1]
        del self.results[-1 - amount:-1]
        return dt


class RecognitionAgent(Agent):
    class MonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref = outer_ref
            self.__loop = asyncio.get_event_loop()
            self.__loop.set_default_executor(executor)

        async def load_model(self):
            print(f"{self.__outer_ref.jid} loading model . . .")
            self.__model = await self.__loop.run_in_executor(None, lambda: RecognitionModel.load_model_from_binary(
                self.__outer_ref.model))
            print(f"{self.__outer_ref.jid} done loading model . . .")

        async def on_start(self):
            await self.load_model()
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")
            global start_time
            if not start_time:
                start_time = time.time()

        async def run(self):
            print(f"{self.__outer_ref.jid} polling. . .")
            data = self.__outer_ref.blackboard.poll(self.__outer_ref.agents_to_respond, 4)
            if len(data) == 0:
                # TODO: Real behavior would be sleeping for a while before polling again
                self.kill()
                return
            print(f"{self.__outer_ref.jid} starting resolving. . .")
            result = await self.__loop.run_in_executor(None, lambda: self.__model.predict_from_faces_images(data))
            self.__outer_ref.blackboard.results.append(result)
            print(f"{self.__outer_ref.jid} done resolving . . .")

        async def on_end(self):
            global end_time, start_time, recog_ag_count
            end_time = time.time()
            recog_ag_count -= 1
            print(end_time - start_time)
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")

    def __init__(self, jid, password, blackboard: Blackboard, agents_to_serve, model, verify_security=False):
        self.jid = jid
        self.model = model
        self.password = password
        self.blackboard = blackboard
        self.agents_to_respond = agents_to_serve
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print("Agent starting . . .")
        b = self.MonitoringBehavior(self)
        self.add_behaviour(b)


class InterfaceServer(multiprocessing.Process):
    def __init__(self, requests: multiprocessing.Queue, responses: multiprocessing.Queue):
        super().__init__()
        self.requests = requests
        self.responses = responses
        self.__loop = None

    async def start_requests_server(self):
        req_server = await asyncio.start_server(
            self.requests_handler, '127.0.0.1', 8888)
        async with req_server:
            await req_server.serve_forever()

    async def requests_handler(self, reader, writer):
        # TODO: Add closing stream logic
        while True:
            print("Processing requests...")
            data_len = int.from_bytes(await reader.read(4), byteorder='big')
            data = await reader.read(data_len)
            self.requests.put(data.decode())

    async def responses_handler(self):
        while True:
            print("Processing responses...")
            r = await self.__loop.run_in_executor(None, lambda: self.responses.get())
            # TODO: Retain the streams and send the answer to the proper stream to the detection agent
            print(f"Sending: {r}")

    def run(self):
        executor = futures.ThreadPoolExecutor(max_workers=4)
        self.__loop = asyncio.get_event_loop()
        self.__loop.set_default_executor(executor)
        try:
            asyncio.ensure_future(self.responses_handler())
            asyncio.ensure_future(self.start_requests_server())
            self.__loop.run_forever()
        except Exception:
            pass
        finally:
            self.__loop.close()


class ControlAgent(Agent):
    class ResultsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref = outer_ref
            self.__loop = asyncio.get_event_loop()
            self.__loop.set_default_executor(executor)

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring results . . .")
            global start_time
            if not start_time:
                start_time = time.time()

        def enqueue_data(self, data):
            for d in data:
                self.__outer_ref.interface_server.responses.put(d)

        async def run(self):
            print(f"{self.__outer_ref.jid} polling for results. . .")
            data = self.__outer_ref.blackboard.poll_results()
            if len(data) == 0:
                if recog_ag_count == 0:
                    self.kill()
                else:
                    await asyncio.sleep(0.5)
            else:
                print(f"{self.__outer_ref.jid} starting resolving results. . .")
                await self.__loop.run_in_executor(None, lambda: self.enqueue_data(data))
                print(f"{self.__outer_ref.jid} done resolving results. . .")

        async def on_end(self):
            global end_time, start_time
            end_time = time.time()
            print(end_time - start_time)
            print(f"{self.__outer_ref.jid} ending monitoring results. . .")

    class RequestsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref = outer_ref
            self.__loop = asyncio.get_event_loop()
            self.__loop.set_default_executor(executor)

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring requests. . .")
            global start_time
            if not start_time:
                start_time = time.time()

        def dequeue_requests(self, amount=-1):
            req = []
            if amount == -1:
                amount = self.__outer_ref.interface_server.requests.qsize()

            try:
                while amount > 0:
                    req.append(self.__outer_ref.interface_server.requests.get_nowait())
                    amount -= 1
            except queue.Empty:
                pass
            finally:
                return req

        async def run(self):
            print(f"{self.__outer_ref.jid} waiting for requests. . .")
            requests = await self.__loop.run_in_executor(None, lambda: self.dequeue_requests())
            if len(requests) == 0:
                if recog_ag_count == 0:
                    self.kill()
                else:
                    await asyncio.sleep(1)
            else:
                print(f"{self.__outer_ref.jid} starting resolving requests. . .")
                # TODO: Add data for agents and retrieve results to send back
                self.__outer_ref.blackboard.results.extend(requests)  # for now just put
                print(f"{self.__outer_ref.jid} done resolving requests. . .")

        async def on_end(self):
            global end_time, start_time
            end_time = time.time()
            print(end_time-start_time)
            print(f"{self.__outer_ref.jid} ending monitoring requests. . .")

    def __init__(self, jid, password, blackboard: Blackboard, interface_server: InterfaceServer, verify_security=False):
        self.jid = jid
        self.blackboard = blackboard
        self.password = password
        self.interface_server = interface_server
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print("Agent starting . . .")
        b1 = self.ResultsMonitoringBehavior(self)
        b2 = self.RequestsMonitoringBehavior(self)
        self.add_behaviour(b1)
        self.add_behaviour(b2)


if __name__ == "__main__":
    executor = futures.ThreadPoolExecutor(max_workers=8)
    responses = multiprocessing.Queue()
    requests = multiprocessing.Queue()
    server = InterfaceServer(requests, responses)
    blackboard = Blackboard()
    recog_ag_count = 5

    ctrl = ControlAgent("robertkarol-ctrl1@404.city", "MeMeS-4TheWin", blackboard, server)
    ag1 = RecognitionAgent("robertkarol-rec1@404.city", "MeMeS-4TheWin", blackboard, 1, model='partial2')
    ag2 = RecognitionAgent("robertkarol-rec2@404.city", "MeMeS-4TheWin", blackboard, 2, model='partial2')
    ag3 = RecognitionAgent("robertkarol-rec3@404.city", "MeMeS-4TheWin", blackboard, 3, model='partial3')
    ag4 = RecognitionAgent("robertkarol-rec4@404.city", "MeMeS-4TheWin", blackboard, 4, model='partial4')
    ag5 = RecognitionAgent("robertkarol-rec5@404.city", "MeMeS-4TheWin", blackboard, 5, model='partial5')

    server.start()
    ctrl.start()
    ag1.start()
    ag2.start()
    ag3.start()
    ag4.start()
    ag5.start()
