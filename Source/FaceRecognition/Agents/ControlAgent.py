import asyncio
import queue
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from Server.InterfaceServer import InterfaceServer


class ControlAgent(Agent):
    class RecognitionResultsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring results . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} polling for results. . .")
            data = self.__outer_ref.blackboard.poll_results(20)
            if len(data) == 0:
                await asyncio.sleep(1)
            else:
                print(f"{self.__outer_ref.jid} starting resolving results. . .")
                await self.__outer_ref.loop.run_in_executor(None,
                                            lambda: self.__outer_ref.interface_server.enqueue_responses(data))
                print(f"{self.__outer_ref.jid} done resolving results. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending monitoring results. . .")

    class RecognitionRequestsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring requests. . .")

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
            requests = await self.__outer_ref.loop.run_in_executor(None,
                                            lambda: self.__outer_ref.interface_server.dequeue_requests(20))
            if len(requests) == 0:
                await asyncio.sleep(1)
            else:
                print(f"{self.__outer_ref.jid} starting resolving requests. . .")
                # TODO: Add data for agents and retrieve results to send back
                print(f"{self.__outer_ref.jid} done resolving requests. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending monitoring requests. . .")

    def __init__(self, jid, password, blackboard: RecognitionBlackboard, interface_server: InterfaceServer,
                 executor, verify_security=False):
        self.jid = jid
        self.blackboard = blackboard
        self.password = password
        self.interface_server = interface_server
        self.loop = asyncio.get_event_loop()
        self.loop.set_default_executor(executor)
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print("Agent starting . . .")
        res_beh = self.RecognitionResultsMonitoringBehavior(self)
        req_beh = self.RecognitionRequestsMonitoringBehavior(self)
        self.add_behaviour(res_beh)
        self.add_behaviour(req_beh)
